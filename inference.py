from model import CycleModel
import data_processing as dp
import json
import random

f = open('json_datasets/train.json', 'r')

raw_train = json.load(f)

vocab = dp.Vocabulary()
vocab.parseText(raw_train)

#for training
cycle_model = CycleModel(vocab)
cycle_model.load_best_model()

f = open('json_datasets/train.json', 'r')
raw_dev = json.load(f)

raw_dev_graphs = random.choices(raw_dev, k = 5).copy()
input_graphs = []

for item in raw_dev_graphs:
	input_graphs.append({'relations' : item['relations'], 'entities' : item['entities']})
                         
raw_dev_text = random.choices(raw_dev, k = 5).copy()
input_text = []

for item in raw_dev_text:
	input_text.append({'text' : item['text'], 'entities' : item['entities']})
                        

print("Input graphs")
print(input_graphs)
g2t_pred = cycle_model.g2t_model.predict(input_graphs, replace_ents=False)
print("Predicted text")
print(g2t_pred)
print()

print("Input text")
print(input_text)
t2g_pred = cycle_model.t2g_model.predict(input_text)
print("Predicted graphs")
print(t2g_pred)
