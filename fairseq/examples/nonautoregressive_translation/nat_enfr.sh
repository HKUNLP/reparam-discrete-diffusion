DATA_TAG=data-bin/wmt14_enfr
ARCH=cmlm_transformer
UPDATE_FREQ=$(( 16 / $ARNOLD_WORKER_GPU ))
CKPT_DIR=cmlm_checkpoints_enfr

# Download and prepare the data
cd examples/translation/
bash prepare-wmt14en2fr.sh
cd ../..

# Binarize the dataset
TEXT=examples/translation/wmt14_en_fr
fairseq-preprocess --joined-dictionary \
    --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $DATA_TAG --thresholdtgt 0 --thresholdsrc 0 \
    --workers 60

python3 -m fairseq_cli.train $DATA_TAG \
    --save-dir $CKPT_DIR \
    --ddp-backend=legacy_ddp \
    --task translation_lev \
    --criterion nat_loss \
    --arch $ARCH \
    --noise random_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 --no-progress-bar \
    --fixed-validation-seed 7 \
    --max-tokens 8192 --update-freq $UPDATE_FREQ \
    --save-interval-updates 10000 \
    --keep-interval-updates 20 \
    --keep-last-epochs 2 \
    --max-update 500000

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
bash scripts/sacrebleu.sh wmt14/full en fr $CKPT_DIR/generate.out