set -ex

# process named arguments
err_msg() { echo "Invalid arguments" 1>&2; exit 1; }

while getopts ":a:c:d:t:e:" o; do
    case "${o}" in
        a)
            AVG_CKPTS=${OPTARG}
            ;;
        c)
            CKPT_DIR=${OPTARG}
            ;;
        d)
            DATASET=${OPTARG}
            ;;
        e)
            # the end of named arguments
            break
            ;;
        *)
            err_msg
            ;;
    esac
done
shift $((OPTIND-1))
CKPT_DIR=${CKPT_DIR:-none}
AVG_CKPTS=${AVG_CKPTS:-true}
DATASET=${DATASET:-iwslt}

if [[ $DATASET == "iwslt" ]]; then
    DATA_TAG=data-bin/iwslt14.tokenized.de-en
    COND="cond"
    DETERMINISTIC="deterministic"
    STRATEGY="cosine"
elif [[ $DATASET == "wmt14" ]]; then
    DATA_TAG=data-bin/wmt14_ende
    COND="uncond"
    DETERMINISTIC="deterministic"
    STRATEGY="cosine"
elif [[ $DATASET == "wmt16" ]]; then
    DATA_TAG=data-bin/wmt16_enro
    COND="uncond"
    DETERMINISTIC="stochastic1.0"
    STRATEGY="cosine"
else
    DATA_TAG=null
fi

TASK=diffusion_translation

if "$AVG_CKPTS"; then
    CKPT_NAME="checkpoint.avg5.pt"
    python3 scripts/average_checkpoints.py \
    --inputs $CKPT_DIR \
    --num-update-checkpoints 5 \
    --output $CKPT_DIR/$CKPT_NAME
else
    CKPT_NAME=${CKPT_DIR##*/}
    CKPT_DIR=${CKPT_DIR%/*}
fi

for NUM_ITER in 2 4 10 16 25
do
    for DECODING_STRATEGY in "--decoding-strategy default" "--decoding-strategy reparam-$COND-$DETERMINISTIC-$STRATEGY"
    do
        fairseq-generate \
            $DATA_TAG \
            --gen-subset test \
            --user-dir diffusion_mt\
            --task $TASK \
            --path $CKPT_DIR/$CKPT_NAME \
            --iter-decode-max-iter $NUM_ITER \
            --iter-decode-eos-penalty 0 \
            --iter-decode-with-beam 5 \
            --iter-decode-force-max-iter \
            --retain-iter-history \
            --argmax-decoding $DECODING_STRATEGY \
            --load-ema-weights\
            --remove-bpe \
            --print-step \
            --batch-size 25 $MODEL_OVERRIDE_ARGS > $CKPT_DIR/generate.out
        echo "###################### NUM_ITER: $NUM_ITER $DECODING_STRATEGY #################################"
        echo "--------------> compound split BLEU <----------------"
        bash scripts/compound_split_bleu.sh $CKPT_DIR/generate.out
        echo "###################### NUM_ITER: $NUM_ITER $DECODING_STRATEGY #################################"
    done
done
