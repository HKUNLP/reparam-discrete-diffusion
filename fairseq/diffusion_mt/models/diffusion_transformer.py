# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements diffusion translation transformers.
"""
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from collections import namedtuple
from diffusion_mt.models.diffusion_transformer_decoder import DiffusionTransformerDecoder
from diffusion_mt.models.diffusion_transformer_encoder import DiffusionTransformerEncoder

from diffusion_mt.time_sampler import UniformSampler, LossSecondMomentResampler, NoneSampler
from discrete_diffusions.absorbing_diffusion import AbsorbingDiffusion
from discrete_diffusions.multinomial_diffusion import MultinomialDiffusion
from discrete_diffusions.reparam_absorbing_diffusion import ReparamAbsorbingDiffusion
from discrete_diffusions.reparam_multinomial_diffusion import ReparamMultinomialDiffusion

import logging

logger = logging.getLogger(__name__)

DecoderOut = namedtuple(
    "DiffusionDecoderOut",
    ["output_tokens", "output_scores", "auxiliary_output", "attn", "step", "max_step", "history"],
)

@register_model("diffusion_transformer")
class DiffusionTransformerModel(FairseqNATModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if self.args.time_sampler_type == "uniform":
            self.time_sampler = UniformSampler(self.args.num_diffusion_timesteps)
        elif self.args.time_sampler_type == "is":
            self.time_sampler = LossSecondMomentResampler(self.args.num_diffusion_timesteps)
        elif self.args.time_sampler_type == "none":
            self.time_sampler = NoneSampler(self.args.num_diffusion_timesteps)
        else:
            self.time_sampler = None
            raise NotImplementedError(f"unknown schedule sampler: {self.args.time_sampler_type}")
        
        if not hasattr(self.args, "not_diffusing_special_sym"):
            if hasattr(self.args, "sampling_masking_sym"):
                self.args.not_diffusing_special_sym = self.args.sampling_masking_sym
            elif hasattr(self.args, "absorbing_masking_sym"):
                self.args.not_diffusing_special_sym = self.args.absorbing_masking_sym
        if not hasattr(self.args, "reweighting_type"):
            self.args.reweighting_type = "linear"
        if not hasattr(self.args, "noise_distribution"):
            self.args.noise_distribution = "uniform"

        pad_id = self.tgt_dict.pad()
        bos_id = self.tgt_dict.bos()
        eos_id = self.tgt_dict.eos()
        if self.args.diffusion_type in ['multinomial']:
            if self.args.label_smoothing > 0.0:
                logger.warning("Enabling label_smooth might lead to worse performance for multinomial diffusion.")
            if not self.args.not_diffusing_special_sym:
                logger.warning("Diffusing special symbols might lead to worse performance for multinomial diffusion.")
        if self.args.diffusion_type == 'absorbing':
            self.diffusion = AbsorbingDiffusion(
                self.args.num_diffusion_timesteps,
                self.tgt_dict.unk(), 
                self.args.lambda_direct_xentropy, 
                self.args.not_diffusing_special_sym,
                pad_id, bos_id, eos_id
            )
        elif self.args.diffusion_type == 'multinomial':
            self.diffusion = MultinomialDiffusion(
                self.args.num_diffusion_timesteps,
                len(self.tgt_dict), 
                self.args.lambda_direct_xentropy, 
                self.args.decoder_loss_type, 
                self.args.noise_scheduler_type,
                self.args.not_diffusing_special_sym,
                pad_id, bos_id, eos_id
            )
        elif self.args.diffusion_type == 'reparam-absorbing':
            self.diffusion = ReparamAbsorbingDiffusion(
                self.args.num_diffusion_timesteps,
                self.tgt_dict.unk(), 
                self.args.reweighting_type,
                self.args.not_diffusing_special_sym,
                pad_id, bos_id, eos_id
            )
        elif self.args.diffusion_type == 'reparam-multinomial':
            vocab_count = self.tgt_dict.count if self.args.noise_distribution == "unigram" else None
            self.diffusion = ReparamMultinomialDiffusion(
                self.args.num_diffusion_timesteps,
                len(self.tgt_dict), 
                self.args.reweighting_type,
                self.args.noise_scheduler_type,
                self.args.not_diffusing_special_sym,
                self.args.noise_distribution,
                pad_id, bos_id, eos_id,
                vocab_count=vocab_count,
            )
        else:
            raise NotImplementedError("Diffusion with type {} is not implemented yet.".format(self.args.diffusion_type))
        
    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        parser.add_argument(
            '--encoder-concat-pe',
            action='store_true',
            default=True,
            help="CMLMC style concating positional-encoding."
        )
        parser.add_argument(
            '--decoder-concat-pe',
            action='store_true',
            default=True,
            help="CMLMC style concating positional-encoding."
        )
        parser.add_argument(
            '--decoder-use-rpe',
            action="store_true",
            default=False,
            help="Use T5-style RPE for Transformer decoders."
        )
        parser.add_argument(
            '--decoder-rpe-num-buckets',
            type=int,
            default=64,
            help="Number of buckets for T5-style RPE in Transformer decoders."
        )
        parser.add_argument(
            '--encoder-use-rpe',
            action="store_true",
            default=False,
            help="Use T5-style RPE for Transformer encoders."
        )
        parser.add_argument(
            '--encoder-rpe-num-buckets',
            type=int,
            default=64,
            help="Number of buckets for T5-style RPE in Transformer encoders."
        )
        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = DiffusionTransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DiffusionTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def _prepare_sample(self, sample, **kwargs):

        # mask all tokens that are pad, bos, or eos.
        non_special_sym_mask = (
            sample["target"].ne(self.tgt_dict.pad()) & 
            sample["target"].ne(self.tgt_dict.bos()) & 
            sample["target"].ne(self.tgt_dict.eos())
        )
        # B x T
        if self.args.q_sample_mode == "default":
            # we use 1 sample for the default sampling trick.
            num_q_samples = 1
            src_tokens = sample["net_input"]["src_tokens"]
            src_lengths = sample["net_input"]["src_lengths"]
            tgt_tokens = sample["target"]
        elif self.args.q_sample_mode in ["coupled", "multi-sample", "multi-step"]:
            # we use 2 samples by default for these advanced sampling tricks,
            # but feel free to specify as you like.
            num_q_samples = 2
            src_tokens = sample["net_input"]["src_tokens"].repeat(num_q_samples, 1)
            src_lengths = sample["net_input"]["src_lengths"].repeat(num_q_samples, 1)
            tgt_tokens = sample["target"].repeat(num_q_samples, 1)

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        if self.args.diffusion_type in ['absorbing', 'reparam-absorbing']:
            # Absorbing diffusion
            if self.args.q_sample_mode == "coupled":
                t1, weight_t = self.time_sampler.sample(sample["target"].shape[0], sample["target"].device)
                t2, _ = self.time_sampler.sample(sample["target"].shape[0], sample["target"].device)
                x_t, x_0_ignore, mask, t = self.diffusion.q_sample_coupled(x_0=sample["target"], t1=t1, t2=t2, non_special_sym_mask=non_special_sym_mask) 
                weight_t = weight_t.repeat(num_q_samples)
            elif self.args.q_sample_mode == "multi-sample":
                rets = []
                t, weight_t = self.time_sampler.sample(sample["target"].shape[0], sample["target"].device)
                for _ in range(num_q_samples):
                    x_t, x_0_ignore, mask = self.diffusion.q_sample(x_0=sample["target"], t=t, non_special_sym_mask=non_special_sym_mask)
                    rets.append((t, weight_t, x_t, x_0_ignore, mask))
                t, weight_t, x_t, x_0_ignore, mask = map(lambda x: torch.cat(x, dim=0), zip(*rets))
            elif self.args.q_sample_mode == "multi-step":
                rets = []
                for _ in range(num_q_samples):
                    t, weight_t = self.time_sampler.sample(sample["target"].shape[0], sample["target"].device)
                    x_t, x_0_ignore, mask = self.diffusion.q_sample(x_0=sample["target"], t=t, non_special_sym_mask=non_special_sym_mask)
                    rets.append((t, weight_t, x_t, x_0_ignore, mask))
                t, weight_t, x_t, x_0_ignore, mask = map(lambda x: torch.cat(x, dim=0), zip(*rets))
            elif self.args.q_sample_mode == "default":
                t, weight_t = self.time_sampler.sample(sample["target"].shape[0], sample["target"].device)
                x_t, x_0_ignore, mask = self.diffusion.q_sample(x_0=sample["target"], t=t, non_special_sym_mask=non_special_sym_mask)
            diffusion_dict = {
                "x_t" : x_t,
                "x_0_ignore" : x_0_ignore,
                "masks" : mask,
                "t": t,
                "weight_t": weight_t
            }
        elif self.args.diffusion_type in ['multinomial', 'reparam-multinomial']:
            if self.args.q_sample_mode == "coupled":
                t1, weight_t = self.time_sampler.sample(sample["target"].shape[0], sample["target"].device)
                t2, _ = self.time_sampler.sample(sample["target"].shape[0], sample["target"].device)
                log_x_t, t = self.diffusion.q_sample_coupled(x_0=sample["target"], t1=t1, t2=t2, non_special_sym_mask=non_special_sym_mask)
                x_t = log_x_t.argmax(dim=-1)
                weight_t = weight_t.repeat(num_q_samples)
                non_special_sym_mask = non_special_sym_mask.repeat(num_q_samples, 1)
            elif self.args.q_sample_mode == "multi-sample":
                t, weight_t = self.time_sampler.sample(sample["target"].shape[0], sample["target"].device)
                rets = []
                for _ in range(num_q_samples):
                    log_x_t = self.diffusion.q_sample(x_0=sample["target"], t=t, non_special_sym_mask=non_special_sym_mask)
                    rets.append((t, weight_t, log_x_t, non_special_sym_mask))
                t, weight_t, log_x_t, non_special_sym_mask = map(lambda x: torch.cat(x, dim=0), zip(*rets))
                x_t = log_x_t.argmax(dim=-1)
            elif self.args.q_sample_mode == "multi-step":
                rets = []
                for _ in range(num_q_samples):
                    t, weight_t = self.time_sampler.sample(sample["target"].shape[0], sample["target"].device)
                    log_x_t = self.diffusion.q_sample(x_0=sample["target"], t=t, non_special_sym_mask=non_special_sym_mask)
                    rets.append((t, weight_t, log_x_t, non_special_sym_mask))
                t, weight_t, log_x_t, non_special_sym_mask = map(lambda x: torch.cat(x, dim=0), zip(*rets))
                x_t = log_x_t.argmax(dim=-1)
            elif self.args.q_sample_mode == "default":
                t, weight_t = self.time_sampler.sample(sample["target"].shape[0], sample["target"].device)
                # alphas in multinomial diffusion are indexed by [0, T).
                log_x_t = self.diffusion.q_sample(x_0=sample["target"], t=t, non_special_sym_mask=non_special_sym_mask)
                x_t = log_x_t.argmax(dim=-1)
            diffusion_dict = {
                "log_x_t" : log_x_t, # [b, n, c]
                "x_t" : x_t,
                "t": t,
                "weight_t": weight_t,
                "non_special_sym_mask": non_special_sym_mask,
            }
        else:
            raise NotImplementedError

        decoder_outputs = self.decoder(
            normalize=False,
            prev_output_tokens=diffusion_dict["x_t"],
            encoder_out=encoder_out,
            t=diffusion_dict["t"],
        ) # a tuple ([B, N, C], None) or ([B, N, C], [B, N])
        diffusion_dict["decoder_outputs"] = decoder_outputs
       
        diffusion_dict["x_0"] = tgt_tokens
        length_dict = {
            "length_out"  : length_out,
            "length_tgt"  : length_tgt,
        }
        return diffusion_dict, length_dict

    def forward(self, sample, **kwargs):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # prepare samples, including sampling time-steps & x_t's.
        diffusion_dict, length_dict = self._prepare_sample(sample, **kwargs)

        ################################################
        # compute diffusion losses
        diffusion_losses, logging_outputs = self.diffusion.compute_loss(
            inputs=diffusion_dict, 
            label_smoothing=self.args.label_smoothing,
        )
        ##################################################
        # if isinstance(self.schedule_sampler, LossAwareSampler):
        #     self.schedule_sampler.update_with_local_losses(
        #         t, losses["squared_kl_step"].detach()
        #     )

        loss_dict = {
            "word_ins": {
                "loss": diffusion_losses["diffusion_loss"],
                "nll_loss": diffusion_losses.get("diffusion_nll_loss", None),
            },
            "length": {
                "out": length_dict["length_out"],
                "tgt": length_dict["length_tgt"],
                "factor": self.decoder.length_loss_factor,
            },
        }
        return loss_dict

    def forward_decoder(self, decoder_out, encoder_out, **kwargs):
        if self.diffusion is None:
            raise NotImplementedError("No diffusion decoding function is provided.")
        def denoising_fn(x_t, t):
            return self.decoder(
                prev_output_tokens=x_t,
                t=t,
                normalize=False,
                encoder_out=encoder_out,
            )
        new_decoder_out = self.diffusion.sample_step(
            decoder_out,
            denoising_fn,
            **kwargs
        )
        return new_decoder_out

    def _initialize_tokens(self, batch_size, dtype, device, idx_length, max_length, length_tgt):
        if self.args.diffusion_type in ['reparam-multinomial']:
            # for multinomial diffusion types,
            # we start with randomly selected tokens.
            if hasattr(self.diffusion, "vocab_size"):
                vocab_size = self.diffusion.vocab_size
            eps = 1e-8
            if False:
                uniform_noise = torch.rand(
                    (batch_size, max_length, vocab_size), 
                    dtype=self.diffusion.vocab_log_prob.dtype,
                    device=device
                )
                gumbel_noise = -torch.log(-torch.log(uniform_noise + eps) + eps)
                initial_output_tokens = (gumbel_noise + self.diffusion.vocab_log_prob).argmax(dim=-1)
            else:
                uniform_noise = torch.rand(
                    (batch_size, vocab_size), 
                    dtype=self.diffusion.vocab_log_prob.dtype,
                    device=device
                )
                gumbel_noise = -torch.log(-torch.log(uniform_noise + eps) + eps)
                (val, ind) = torch.topk(gumbel_noise + self.diffusion.vocab_log_prob, k=max_length)
                initial_output_tokens = ind.gather(-1, torch.rand_like(ind.float()).argsort(-1))
            initial_output_tokens.masked_fill_(
                idx_length[None, :] >= length_tgt[:, None], self.pad
            )
        elif self.args.diffusion_type in ['multinomial']:
            # for multinomial diffusion types,
            # we start with randomly selected tokens.
            if hasattr(self.diffusion, "vocab_size"):
                vocab_size = self.diffusion.vocab_size
            initial_output_tokens = torch.randint(
                0,
                vocab_size, 
                size=(batch_size, max_length),
                dtype=dtype,
                device=device)
            initial_output_tokens.masked_fill_(
                idx_length[None, :] >= length_tgt[:, None], self.pad
            )
        elif self.args.diffusion_type in ['absorbing', 'reparam-absorbing']:
            # for masking diffusion types,
            # we start with a whole [M] sequence.
            initial_output_tokens = length_tgt.new_zeros(
                batch_size, max_length
            ).fill_(self.pad)
            initial_output_tokens.masked_fill_(
                idx_length[None, :] < length_tgt[:, None], self.unk
            )
        else:
            raise NotImplementedError
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        return initial_output_tokens
    
    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )
        # <bos>, <eos>, and at least one token.
        # max_length = length_tgt.clamp_(min=2).max()
        max_length = length_tgt.clamp_(min=3).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = self._initialize_tokens(
            src_tokens.size(0),
            src_tokens.dtype, 
            src_tokens.device,
            idx_length,
            max_length,
            length_tgt
        )
        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        initial_output_masks = (
            initial_output_tokens.ne(self.tgt_dict.pad()) & 
            initial_output_tokens.ne(self.tgt_dict.bos()) & 
            initial_output_tokens.ne(self.tgt_dict.eos())
        )
        

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            auxiliary_output={
                "output_masks" : initial_output_masks,
            },
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, length_beam_size, length_within_beam=1):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, length_beam_size)
            - length_beam_size // 2
        ).repeat(1, length_within_beam)
        # <bos>, <eos>, and at least one token.
        # length_tgt = length_tgt.view(-1).clamp_(min=2)
        length_tgt = length_tgt.view(-1).clamp_(min=3)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = self._initialize_tokens(
            length_tgt.size(0),
            output_tokens.dtype, 
            output_tokens.device,
            idx_length,
            max_length,
            length_tgt
        )

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        initial_output_masks = (
            initial_output_tokens.ne(self.tgt_dict.pad()) & 
            initial_output_tokens.ne(self.tgt_dict.bos()) & 
            initial_output_tokens.ne(self.tgt_dict.eos())
        )

        return decoder_out._replace(
            output_tokens=initial_output_tokens, 
            output_scores=initial_output_scores,
            auxiliary_output={
                "output_masks" : initial_output_masks,
            },
        )

@register_model_architecture("diffusion_transformer", "diffusion_transformer")
def diffusion_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture("diffusion_transformer", "diffusion_transformer_iwslt")
def diffusion_iwslt(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    diffusion_base_architecture(args)

@register_model_architecture("diffusion_transformer", "diffusion_transformer_wmt")
def diffusion_wmt(args):
    diffusion_base_architecture(args)


@register_model_architecture("diffusion_transformer", "diffusion_transformer_diffuseq")
def diffusion_diffuseq(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    diffusion_base_architecture(args)

@register_model_architecture("diffusion_transformer", "diffusion_transformer_qg")
@register_model_architecture("diffusion_transformer", "diffusion_transformer_qqp")
def diffusion_qqp_qg(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    diffusion_base_architecture(args)