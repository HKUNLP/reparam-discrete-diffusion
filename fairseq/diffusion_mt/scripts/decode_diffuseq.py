#!/usr/bin/env python3 -u
# adapted from fairseq.generate
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
import json
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.file_io import PathManager
from transformers import BertTokenizer
def _load_model_ensemble_with_ema(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    state=None,
):
    """Loads an ensemble of models.

    Args:
        filenames (List[str]): checkpoint files to load
        arg_overrides (Dict[str,Any], optional): override model args that
            were used during model training
        task (fairseq.tasks.FairseqTask, optional): task to use for loading
    """
    assert state is None or len(filenames) == 1

    from fairseq import tasks

    ensemble = []
    cfg = None
    for filename in filenames:
        filename = filename.replace(".pt", suffix + ".pt")

        if not PathManager.exists(filename):
            raise IOError("Model file not found: {}".format(filename))
        if state is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)
        if "args" in state and state["args"] is not None:
            cfg = convert_namespace_to_omegaconf(state["args"])
        elif "cfg" in state and state["cfg"] is not None:
            cfg = state["cfg"]
        else:
            raise RuntimeError(
                f"Neither args nor cfg exist in state keys = {state.keys()}"
            )

        if task is None:
            task = tasks.setup_task(cfg.task)

        if "task_state" in state:
            task.load_state_dict(state["task_state"])

        # model parallel checkpoint or unsharded checkpoint
        model = task.build_model(cfg.model)
        if (
            "optimizer_history" in state
            and len(state["optimizer_history"]) > 0
            and "num_updates" in state["optimizer_history"][-1]
        ):
            model.set_num_updates(state["optimizer_history"][-1]["num_updates"])

        # if self.cfg.ema.ema_fp32:
        #     # use EMA params in fp32
        #     state_dict["extra_state"]["ema_fp32_params"]
        assert "extra_state" in state and "ema" in state["extra_state"]
        model.load_state_dict(
            state["extra_state"]["ema"], strict=strict, model_cfg=cfg.model
        )

        # reset state so it gets loaded for the next model in ensemble
        state = None

        # build model for ensemble
        ensemble.append(model)
    return ensemble, cfg

def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)

def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    if cfg.task.load_ema_weights and cfg.checkpoint.checkpoint_shard_count == 1:
        models, saved_cfg = _load_model_ensemble_with_ema(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        )
    else:
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    lms = [None]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs, task_args=cfg.task
    )

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()

    if cfg.common_eval.results_path is not None:
        json_output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.json".format(cfg.dataset.gen_subset),
        )
        json_fout = open(json_output_path, 'w', encoding="utf-8")

    
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        gen_timer.start()
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=None,
            constraints=None,
        ) # [[], [], []]
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample["id"].tolist()):
            has_target = sample["target"] is not None

            def strip_fairseq_special_syms(tensor):
                return tensor[
                    tensor.ne(tgt_dict.pad()) & 
                    tensor.ne(tgt_dict.bos()) &
                    tensor.ne(tgt_dict.eos())
                ]
            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = strip_fairseq_special_syms(
                    sample["net_input"]["src_tokens"][i, :]
                )
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = (
                    strip_fairseq_special_syms(sample["target"][i, :])
                    .int()
                    .cpu()
                )
            
            def hf_tokenizer_decode(tokens):
                shifted_ids = tokens - 4
                shifted_ids = shifted_ids.masked_fill(shifted_ids < 0, tokenizer.mask_token_id)
                return tokenizer.decode(shifted_ids, skip_special_tokens=True)

            if src_dict is not None:
                src_str = hf_tokenizer_decode(src_tokens)
            else:
                src_str = ""
            if has_target:
                target_str = hf_tokenizer_decode(target_tokens)

            if not cfg.common_eval.quiet:
                if src_dict is not None:
                    print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                if has_target:
                    print("T-{}\t{}".format(sample_id, target_str), file=output_file)

            # Process top predictions
            hypo_strs = []
            for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                stripped_hypo_tokens = strip_fairseq_special_syms(hypo["tokens"].int().cpu())
                hypo_str = hf_tokenizer_decode(stripped_hypo_tokens)
                hypo_strs.append(hypo_str)
                if not cfg.common_eval.quiet:
                    print(
                        "H-{}\t{}\t{}".format(sample_id, j, hypo_str),
                        file=output_file,
                    )

                    if cfg.generation.print_step:
                        print(
                            "I-{}\t{}".format(sample_id, hypo["steps"]),
                            file=output_file,
                        )

                    if cfg.generation.retain_iter_history:
                        for step, h in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                "E-{}_{}\t{}".format(sample_id, step, h_str),
                                file=output_file,
                            )
            
            res_dict = {
                "recover": hypo_strs,
                "ref": target_str,
                "src": src_str,
            }
            print(json.dumps(res_dict), file=json_fout)
        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )
    logger.info(
        "Translated {:,} sentences ({:,} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    json_fout.close()


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
