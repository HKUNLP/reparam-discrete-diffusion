from transformers import BertTokenizer
import os
import json
import argparse

def encode_bpe(data_dir, output_dir, split='train'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer.save_pretrained(args.checkpoint_path)
    path = '{}/{}.jsonl'.format(data_dir, split)
    
    path_src = '{}/{}.bpe.src'.format(output_dir, split)
    path_tgt = '{}/{}.bpe.tgt'.format(output_dir, split)
    with open(path, 'r') as f_reader:
        with open(path_src, 'w', encoding='utf-8') as f_src:
            with open(path_tgt, 'w', encoding='utf-8') as f_tgt:
                for json_l in f_reader:
                    row = json.loads(json_l)
                    src = row['src'].strip()
                    tgt = row['trg'].strip()

                    src_tokens = tokenizer.tokenize(src)
                    tgt_tokens = tokenizer.tokenize(tgt)
                    # manually truncate to align with diffuseq's implementation
                    while len(src_tokens) + len(tgt_tokens) > args.max_seq_len - 5:
                        if len(src_tokens) > len(tgt_tokens):
                            src_tokens.pop()
                        elif len(src_tokens) < len(tgt_tokens):
                            tgt_tokens.pop()
                        else:
                            src_tokens.pop()
                            tgt_tokens.pop()
                    bpe_src = " ".join(src_tokens)
                    bpe_tgt = " ".join(tgt_tokens)
                    f_src.write('{}\n'.format(bpe_src))
                    f_tgt.write('{}\n'.format(bpe_tgt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='data')
    parser.add_argument('--max-seq-len', type=int, default=128)
    args = parser.parse_args()
    encode_bpe(args.data_dir, args.output_dir, 'train')
    encode_bpe(args.data_dir, args.output_dir, 'valid')
    encode_bpe(args.data_dir, args.output_dir, 'test')
