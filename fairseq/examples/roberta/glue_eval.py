from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'roberta.large.mnli/',
    checkpoint_file='model.pt',
    data_name_or_path='MNLI-bin'
)

# label_fn = lambda label: roberta.task.label_dictionary.string(
#     [label + roberta.task.label_dictionary.nspecial]
# )
label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
############ RTE
# with open('glue_data/RTE/dev.tsv') as fin:
#     fin.readline()
#     for index, line in enumerate(fin):
#         tokens = line.strip().split('\t')
#         sent1, sent2, target = tokens[1], tokens[2], tokens[3]
#         tokens = roberta.encode(sent1, sent2)
#         prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
#         prediction_label = label_fn(prediction)
#         ncorrect += int(prediction_label == target)
#         nsamples += 1
# print('| Accuracy: ', float(ncorrect)/float(nsamples))

############ MNLI
# print(roberta)
# with open('glue_data/MNLI/dev_matched.tsv') as fin:
#     fin.readline()
#     for index, line in enumerate(fin):
#         tokens = line.strip().split('\t')
#         sent1, sent2, target = tokens[8], tokens[9], tokens[15]
#         tokens = roberta.encode(sent1, sent2)
#         prediction = roberta.predict('mnli', tokens).argmax().item()
#         prediction_label = label_fn(prediction)
#         ncorrect += int(prediction_label == target)
#         nsamples += 1
# print('| Accuracy: ', float(ncorrect)/float(nsamples))

############ MRPC
# from fairseq.models.roberta import RobertaModel

# roberta = RobertaModel.from_pretrained(
#     'checkpoints/',
#     checkpoint_file='checkpoint_best.pt',
#     data_name_or_path='MRPC-bin'
# )

# label_fn = lambda label: roberta.task.label_dictionary.string(
#     [label + roberta.task.label_dictionary.nspecial]
# )
# ncorrect, nsamples = 0, 0
# roberta.cuda()
# roberta.eval()
# with open('glue_data/MRPC/dev.tsv') as fin:
#     fin.readline()
#     for index, line in enumerate(fin):
#         tokens = line.strip().split('\t')
#         sent1, sent2, target = tokens[3], tokens[4], tokens[0]
#         tokens = roberta.encode(sent1, sent2)
#         prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
#         prediction_label = label_fn(prediction)
#         ncorrect += int(prediction_label == target)
#         nsamples += 1
# print('| Accuracy: ', float(ncorrect)/float(nsamples))