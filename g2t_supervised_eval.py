#Imports
from collections import Counter
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import tqdm
import json
import re
import pandas as pd
from simplet5 import SimpleT5

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def removeQuotes(lst):
    ret = []
    for s in lst:
        if s != '``' and s != "''":
            ret.append(s)
    return ret

def camelCaseSplit(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    d = [m.group(0) for m in matches]
    new_d = []
    for token in d:
        token = token.replace('(', '')
        token = token.replace(')', '')
        token_split = token.split('_')
        for t in token_split:
            #new_d.append(t.lower())
            new_d.append(t)
    return new_d

def g2tPreprocess(raw):
    df = []
    for item in raw:
        graph = 'g2t:'
        for relation in item['relations']:
            graph += ' <H> ' + ' '.join(removeQuotes(relation[0])) + ' <R> '
            graph += ' '.join(camelCaseSplit(relation[1])) + ' <T> '
            graph += ' '.join(removeQuotes(relation[2]))

        ents = [' '.join(removeQuotes(entity)) for entity in item['entities']]
        text = item['text']
        for i in range(len(ents)):
            text = text.replace('<ENT_'+str(i)+'>', ents[i])
        sample = [graph, text]
        df.append(sample)
    return pd.DataFrame(df, columns=['source_text', 'target_text'])


f_test = open('json_datasets/test.json', 'r')
raw_test = json.load(f_test)
f_test.close()
test_df = g2tPreprocess(raw_test)

# load the model
model = SimpleT5()
model.from_pretrained("t5","t5-base")
model.load_model("t5","outputs/simplet5-epoch-4-train-loss-0.3465", use_gpu=True)

ref = defaultdict(list)
ptr = 0
same = defaultdict(list)

for i in range(len(test_df)):
    if i > 0 and test_df['source_text'][i] != test_df['source_text'][i-1]:
        ptr += 1
    same[ptr].append(test_df['target_text'][i].lower())
    ref[i] = same[ptr]

bleu = Bleu(4)
meteor = Meteor()
rouge = Rouge()
cider = Cider()

hyp = [model.predict(test_df['source_text'][i])[0] for i in range(len(test_df))]

hyp = dict(zip(range(len(test_df)), [[x.lower()] for x in hyp]))
# ref = dict(zip(range(len(dev_df)), [[dev_df['target_text'][i]] for i in range(len(dev_df))]))
ret = bleu.compute_score(ref, hyp)
print('BLEU INP {0:}'.format(len(hyp)))
print('BLEU 1-4 {0:}'.format(ret[0]))
print('METEOR {0:}'.format(meteor.compute_score(ref, hyp)[0]))
print('ROUGE_L {0:}'.format(rouge.compute_score(ref, hyp)[0]))
print('Cider {0:}'.format(cider.compute_score(ref, hyp)[0]))



