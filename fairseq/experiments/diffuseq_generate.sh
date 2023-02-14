set -ex
err_msg() { echo "Invalid arguments" 1>&2; exit 1; }

while getopts ":a:b:c:d:t:e:" o; do
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
        b)
            USE_BEAM=${OPTARG}
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
USE_BEAM=${USE_BEAM:-true}

if [[ $DATASET == "qg" ]]; then
    DATA_TAG=data-bin/QG
elif [[ $DATASET == "qqp" ]]; then
    DATA_TAG=data-bin/QQP
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

if "$USE_BEAM"; then
    BEAM=3
    LENGTH_SIZE=3
else
    BEAM=1
    LENGTH_SIZE=1
fi
TOTAL_BEAM_SIZE=$((BEAM * LENGTH_SIZE))

COND="uncond"
DETERMINISTIC="stochastic5.0"
STRATEGY="cosine"
for NUM_ITER in 2 5 10 20 25
do
    for DECODING_STRATEGY in "--decoding-strategy default" "--decoding-strategy reparam-$COND-$DETERMINISTIC-$STRATEGY"
    do
        python diffusion_mt/scripts/decode_diffuseq.py  \
            $DATA_TAG \
            --gen-subset test \
            --user-dir diffusion_mt\
            --task $TASK \
            --path $CKPT_DIR/$CKPT_NAME \
            --results-path $CKPT_DIR \
            --iter-decode-max-iter $NUM_ITER \
            --iter-decode-eos-penalty 0 \
            --iter-decode-with-beam $BEAM \
            --beam-within-length $LENGTH_SIZE \
            --iter-decode-force-max-iter \
            --retain-iter-history \
            --return-all-cands --nbest $TOTAL_BEAM_SIZE \
            --temperature 0.3 $DECODING_STRATEGY \
            --load-ema-weights\
            --remove-bpe \
            --print-step \
            --batch-size 10
        echo "###################### NUM_ITER: $NUM_ITER $TEMP $DECODING_STRATEGY #################################"
        python diffusion_mt/scripts/eval_diffuseq.py --folder $CKPT_DIR
        echo "###################### NUM_ITER: $NUM_ITER $TEMP $DECODING_STRATEGY #################################"
    done
done