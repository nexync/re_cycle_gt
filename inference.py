from model import CycleModel
import data_processing as dp
import json

f = open('json_datasets/train.json', 'r')

raw_train = json.load(f)

vocab = dp.Vocabulary()
vocab.parseText(raw_train)

#for training
cycle_model = CycleModel(vocab)
cycle_model.load_best_model()

f = open('json_datasets/dev.json', 'r')
raw_dev = json.load(f)

g2t_pred = cycle_model.g2t_model.predict(raw_dev[0:8], replace_ents=False)
t2g_pred_matrices = cycle_model.t2g_model.predict(raw_dev[0:8])
t2g_pred = cycle_model.g2t_model.g2t_preprocess(t2g_pred_matrices, mode="TGT")