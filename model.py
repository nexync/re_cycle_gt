# import
from simplet5 import SimpleT5

#Imports

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import tqdm

import data_processing as dp
import json

import t2g
import g2t

# instantiate models
g2t_model = SimpleT5()
t2g_model = t2g.T2GModel()

# load (supports t5, mt5, byT5 models)
g2t_model.from_pretrained("t5","t5-base")


# Opening JSON file
f = open('json_datasets/train.json', 'r')

raw_train = json.load(f)

vocab = dp.Vocabulary()
vocab.parseText(raw_train)

#create cycle

cycle_model = CycleModel(t2g_model, g2t_model)
dataloader = None
cycle_model.train(10, dataloader)


#create dataloader
#t, g = dp.create_cycle_dataloader(vocab, batch_size = 8, shuffle=True)


class CycleModel():
	def __init__(self, vocab):
		self.t2g_model = t2g.T2GModel(vocab, 768)
		self.g2t_model = g2t.G2TModel(vocab)
		self.t2g_opt = torch.optim.Adam(self.t2g_model.model.parameters())
		self.g2t_opt = torch.optim.Adam(self.g2t_model.t5_model.model.parameters())
		self.vocab = vocab
    
	def t_cycle(self, text_batch): # optimizes g2t
		self.t2g_model.eval()
		self.g2t_model.train()
		with torch.no_grad():
			pred_graphs = self.t2g_model.predict(text_batch)
		# syn_batch???
		self.g2t_opt.zero_grad()
		pred_text = self.g2t_model.predict(pred_graphs)   #note: this would not be predict here - it would be calling running through the model i think
		# convert pred_text to tensor of word indices
		loss = F.nll_loss(pred_text.reshape(-1, pred_text.shape[-1]), text_batch.reshape(-1), ignore_index=0) # could be wrong, again
		loss.backward()
		#nn.utils.clip_grad_norm_(g2t_model.parameters(), config['clip'])
		self.g2t_opt.step()
		return loss.item()

	def g_cycle(self, graph_batch): # optimizes t2g
		self.g2t_model.eval()
		self.t2g_model.train()
		with torch.no_grad():
			pred_text = self.g2t_model.predict(graph_batch)
		# convert pred_text to correct format to input into t2g
		self.t2g_opt.zero_grad()
		pred_graphs = self.t2g_model.predict(pred_text)
		loss = F.nll_loss(pred_graphs.contiguous().view(-1, pred_graphs.shape[-1]), graph_batch.contiguous().view(-1), ignore_index=0) # could be wrong, again
		loss.backward()
		#nn.utils.clip_grad_norm_(g2t_model.parameters(), config['clip'])
		self.t2g_opt.step()
		return loss.item()

	def back_translation(self, text_batch, graph_batch):
		g_loss = self.g_cycle(graph_batch)
		t_loss = self.t_cycle(text_batch)
		return g_loss, t_loss

	def train(self, epochs):

		for i in range(epochs):
			tcycle_dataloader, gcycle_dataloader = dp.create_cycle_dataloader(vocab, batch_size = 8, shuffle=True)
			
			with tqdm.tqdm(dataloader) as tqb:
				for i, (text_batch, graph_batch) in enumerate(tqb):
					# need pairings of text/graph batches (unparallel)
					g_loss, t_loss = self.back_translation(text_batch, graph_batch)
                    

    
    