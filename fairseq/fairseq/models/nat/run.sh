wget http://dl.fbaipublicfiles.com/nat/original_dataset.zip
unzip original_dataset.zip
TEXT=../wmt14_ende
fairseq-preprocess --joined-dictionary \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.en-de --validpref $TEXT/valid.en-de --testpref $TEXT/test.en-de \
    --destdir data-bin/wmt14_ende --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

# cd fairseq/examples/translation/
# bash prepare-iwslt14.sh
# cd ../..

# # Preprocess/binarize the data
# TEXT=examples/translation/iwslt14.tokenized.de-en
# fairseq-preprocess --source-lang de --target-lang en \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/iwslt14.tokenized.de-en \
#     --workers 20
DATA_TAG=data-bin/wmt14_ende
ARCH=cmlm_transformer
UPDATE_FREQ=$(( 16 / $ARNOLD_WORKER_GPU ))
python3 -m fairseq_cli.train $DATA_TAG \
    --save-dir cmlm_checkpoints \
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
    --max-update 300000

                  
# fairseq-generate \
#     data-bin/wmt14_en_de_distill \
#     --gen-subset test \
#     --task translation_lev \
#     --path checkpoint_1.pt:checkpoint_2.pt:checkpoint_3.pt \
#     --iter-decode-max-iter 9 \
#     --iter-decode-eos-penalty 0 \
#     --beam 1 --remove-bpe \
#     --print-step \
#     --batch-size 400