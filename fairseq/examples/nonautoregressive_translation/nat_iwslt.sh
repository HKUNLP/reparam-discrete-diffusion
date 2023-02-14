set -ex
# cd examples/translation/
# bash prepare-iwslt14.sh
# cd ../..

# # Preprocess/binarize the data
# TEXT=examples/translation/iwslt14.tokenized.de-en
# fairseq-preprocess --joined-dictionary --source-lang de --target-lang en \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/iwslt14.tokenized.de-en \
#     --workers 20
DATA_TAG=data-bin/iwslt14.tokenized.de-en
if [[ $1 == "cmlm" ]]; then
    UPDATE_FREQ=1
    CKPT_DIR=cmlm_checkpoints
    python3 -m fairseq_cli.train $DATA_TAG \
        --save-dir $CKPT_DIR \
        --ddp-backend=legacy_ddp \
        --task translation_lev \
        --criterion nat_loss \
        --arch cmlm_transformer \
        --noise random_mask \
        --share-all-embeddings \
        --optimizer adam --adam-betas '(0.9,0.98)' \
        --lr 0.0003 --lr-scheduler inverse_sqrt \
        --stop-min-lr '1e-09' --warmup-updates 10000 \
        --warmup-init-lr '1e-07' --label-smoothing 0.1 \
        --dropout 0.3 --weight-decay 0.01 \
        --eval-bleu \
        --decoder-learned-pos \
        --encoder-learned-pos \
        --apply-bert-init \
        --log-format 'simple' --log-interval 100 --no-progress-bar \
        --fixed-validation-seed 7 \
        --max-tokens 2048 --update-freq $UPDATE_FREQ \
        --save-interval-updates 10000 \
        --keep-interval-updates 20 \
        --max-update 250000
elif [[ "$1" == "diffusion" ]]; then
    UPDATE_FREQ=1
    CKPT_DIR=diffusion_checkpoints
    python3 -m fairseq_cli.train $DATA_TAG \
        --save-dir $CKPT_DIR \
        --ddp-backend=legacy_ddp \
        --task diffusion_translation \
        --criterion diffusion_loss \
        --arch diffusion_transformer \
        --share-all-embeddings \
        --optimizer adam --adam-betas '(0.9,0.98)' \
        --lr 0.0003 --lr-scheduler inverse_sqrt \
        --stop-min-lr '1e-09' --warmup-updates 10000 \
        --warmup-init-lr '1e-07' --label-smoothing 0.1 \
        --dropout 0.3 --weight-decay 0.01 \
        --eval-bleu \
        --decoder-learned-pos \
        --encoder-learned-pos \
        --apply-bert-init \
        --log-format 'simple' --log-interval 100 --no-progress-bar \
        --fixed-validation-seed 7 \
        --max-tokens 2048 --update-freq $UPDATE_FREQ \
        --save-interval-updates 10000 \
        --keep-interval-updates 20 \
        --max-update 250000 --user-dir diffusion_mt 
fi

python3 scripts/average_checkpoints.py \
--inputs $CKPT_DIR \
--num-update-checkpoints 5 \
--output $CKPT_DIR/checkpoint.avg5.pt

fairseq-generate \
    $DATA_TAG \
    --gen-subset test \
    --task translation_lev \
    --path $CKPT_DIR/checkpoint.avg5.pt \
    --iter-decode-max-iter 10 \
    --iter-decode-eos-penalty 0 \
    --iter-decode-with-beam 5 \
    --remove-bpe \
    --print-step \
    --batch-size 50 > $CKPT_DIR/generate.out

echo "--------------> compound split BLEU <----------------"
bash scripts/compound_split_bleu.sh $CKPT_DIR/generate.out

echo "--------------> detokenized BLEU with sacrebleu <----------------"
bash scripts/sacrebleu.sh wmt14/full en de $CKPT_DIR/generate.out