set -ex

# process named arguments
err_msg() { echo "Invalid arguments" 1>&2; exit 1; }

while getopts ":m:s:g:d:t:e:" o; do
    case "${o}" in
        m)
            MODEL=${OPTARG}
            ;;
        s)
            SUFFIX=${OPTARG}
            ;;
        d)
            DATASET=${OPTARG}
            ;;
        g)
            # only do the inference
            GENERATE_ONLY=${OPTARG}
            ;;
        t)
            TRAIN_ONLY=${OPTARG}
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
MODEL=${MODEL:-absorbing-diffusion}
SUFFIX=${SUFFIX:-''}
GENERATE_ONLY=${GENERATE_ONLY:-false}
TRAIN_ONLY=${TRAIN_ONLY:-false}
DATASET=${DATASET:-iwslt}

if [[ $DATASET == "iwslt" ]]; then
    DATA_TAG=data-bin/iwslt14.tokenized.de-en
    ARCH=diffusion_transformer_iwslt
    DATA_SPECIFIC_ARGS="--warmup-updates 30000 --max-update 300000 --max-tokens 4096 --update-freq 1"
elif [[ $DATASET == "wmt14" ]]; then
    DATA_TAG=data-bin/wmt14_ende
    ARCH=diffusion_transformer_wmt
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk 'BEGIN{FS=","};{print NF}')
    UPDATE_FREQ=$(( 32 / $NUM_GPUS )) # maintain ~128k tokens
    DATA_SPECIFIC_ARGS="--warmup-updates 30000 --max-update 300000 --max-tokens 4096 --update-freq $UPDATE_FREQ --dropout 0.2"
elif [[ $DATASET == "wmt16" ]]; then
    DATA_TAG=data-bin/wmt16_enro
    ARCH=diffusion_transformer_wmt
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk 'BEGIN{FS=","};{print NF}')
    UPDATE_FREQ=$(( 16 / $NUM_GPUS )) # maintain ~32k tokens
    DATA_SPECIFIC_ARGS="--warmup-updates 15000 --max-update 120000 --max-tokens 2048 --update-freq $UPDATE_FREQ"
else
    DATA_TAG=null
fi

TASK=diffusion_translation
CRITERION=diffusion_loss
CKPT_DIR=checkpoints/$DATASET"_"$MODEL"_checkpoints_"$SUFFIX
SPECIFIC_ARGS="
    --user-dir diffusion_mt --num-diffusion-timesteps 50 --diffusion-type $MODEL $@
    "

if ! "$GENERATE_ONLY"; then
    python3 -m fairseq_cli.train $DATA_TAG \
        --save-dir $CKPT_DIR \
        --ddp-backend=legacy_ddp \
        --task $TASK \
        --criterion $CRITERION \
        --arch $ARCH \
        --share-all-embeddings \
        --optimizer adam --adam-betas '(0.9,0.98)' \
        --lr 0.0005 --stop-min-lr '1e-09' \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
        --label-smoothing 0.1 \
        --dropout 0.3 --weight-decay 0.01 \
        --eval-bleu \
        --eval-bleu-args '{"iter_decode_max_iter": 10, "iter_decode_force_max_iter": true, "temperature": 0.1}' \
        --decoder-learned-pos \
        --encoder-learned-pos \
        --apply-bert-init \
        --log-format 'simple' --log-interval 100 --no-progress-bar \
        --fixed-validation-seed 7 \
        --save-interval-updates 10000 \
        --keep-interval-updates 10 \
        --keep-last-epochs 2 \
        $DATA_SPECIFIC_ARGS $SPECIFIC_ARGS
fi

if ! "$TRAIN_ONLY"; then
    bash experiments/mt_generate.sh -d $DATASET -a true -c $CKPT_DIR
fi