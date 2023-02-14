set -ex
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
MODEL=${MODEL:-absorbing}
SUFFIX=${SUFFIX:-''}
GENERATE_ONLY=${GENERATE_ONLY:-false}
TRAIN_ONLY=${TRAIN_ONLY:-false}
DATASET=${DATASET:-iwslt}

if [[ $DATASET == "qg" ]]; then
    DATA_TAG=data-bin/QG
    ARCH=diffusion_transformer_qg
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk 'BEGIN{FS=","};{print NF}')
    UPDATE_FREQ=$(( 4 / $NUM_GPUS ))
    DATA_SPECIFIC_ARGS="--warmup-updates 10000 --lr 0.0005 --max-update 70000 --max-sentences 64 --dropout 0.2 --update-freq $UPDATE_FREQ"
elif [[ $DATASET == "qqp" ]]; then
    DATA_TAG=data-bin/QQP
    ARCH=diffusion_transformer_qqp
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk 'BEGIN{FS=","};{print NF}')
    UPDATE_FREQ=$(( 4 / $NUM_GPUS ))
    DATA_SPECIFIC_ARGS="--warmup-updates 10000 --lr 0.0005 --max-update 70000 --max-sentences 64 --dropout 0.2 --update-freq $UPDATE_FREQ"
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
        --arch $ARCH \
        --criterion $CRITERION \
        --source-lang src --target-lang tgt \
        --max-source-positions 128 \
        --max-target-positions 128 \
        --decoder-learned-pos \
        --encoder-learned-pos \
        --share-all-embeddings \
        --label-smoothing 0.1 \
        --dropout 0.2 --attention-dropout 0.1 --activation-dropout 0.1 \
        --clip-norm 1.0 \
        --eval-bleu \
        --eval-bleu-args '{"iter_decode_max_iter": 5, "iter_decode_force_max_iter": true, "temperature": 0.1}' \
        --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
        --lr-scheduler inverse_sqrt \
        --find-unused-parameters \
        --skip-invalid-size-inputs-valid-test \
        --log-format 'simple' --log-interval 100 --no-progress-bar \
        --fixed-validation-seed 7 \
        --save-interval-updates 5000 \
        --validate-interval-updates 2000 \
        --validate-interval 10000 \
        --keep-interval-updates 5 \
        --keep-last-epochs 1 \
        $DATA_SPECIFIC_ARGS $SPECIFIC_ARGS
fi

if ! "$TRAIN_ONLY"; then
    bash experiments/diffuseq_generate.sh -d $DATASET -a true -c $CKPT_DIR -b true
fi