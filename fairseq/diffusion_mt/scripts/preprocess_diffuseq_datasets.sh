# fetch raw json datasets from https://github.com/Shark-NLP/DiffuSeq and 
# extract them to diffuseq_data/$TASK, for TASK in {QG, QQP}
TASK=$1
mkdir -p diffuseq_data/bpes/$TASK
python diffusion_mt/scripts/bpe_encode.py --data-dir diffuseq_data/$TASK --output-dir diffuseq_data/bpes/$TASK

fairseq-preprocess \
--source-lang src --target-lang tgt \
--trainpref "diffuseq_data/bpes/${TASK}/train.bpe" \
--validpref "diffuseq_data/bpes/${TASK}/valid.bpe" \
--testpref "diffuseq_data/bpes/${TASK}/test.bpe" \
--destdir "data-bin/${TASK}" --srcdict diffusion_mt/vocab.txt --tgtdict diffusion_mt/vocab.txt \
--workers 20