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
# import g2t
import g2t_copy as g2t

# # instantiate models
# g2t_model = SimpleT5()
# t2g_model = t2g.T2GModel()

# # load (supports t5, mt5, byT5 models)
# g2t_model.from_pretrained("t5","t5-base")


# Opening JSON file
f = open('json_datasets/train.json', 'r')

raw_train = json.load(f)

vocab = dp.Vocabulary()
vocab.parseText(raw_train)

#create cycle

cycle_model = CycleModel(vocab)
dataloader = None
cycle_model.train(10, dataloader)


#create dataloader
#t, g = dp.create_cycle_dataloader(vocab, batch_size = 8, shuffle=True)


class CycleModel():
	def __init__(self, vocab, device = "cpu"):
		if device == "gpu":
			self.device = torch.device('cuda:0')
		else:
			self.device = torch.device('cpu')

		self.t2g_model = t2g.T2GModel(vocab, self.device, 768)
		self.g2t_model = g2t.G2TModel(vocab)
		self.t2g_opt = torch.optim.Adam(self.t2g_model.model.parameters())
		self.g2t_opt = torch.optim.Adam(self.g2t_model.t5_model.parameters())
		self.vocab = vocab
    
	def t_cycle(self, text_batch): # optimizes g2t
		self.t2g_model.eval()
		self.g2t_model.train()

		gold_text, _ = self.t2g_model.t2g_preprocess(text_batch, mode = "TGT").to(self.device) # bs x gold_text_len
		bs, gold_text_len = gold_text.shape

		with torch.no_grad():
			pred_graphs = self.t2g_model.predict(text_batch) #synthetic batch of graphs
		
		self.g2t_opt.zero_grad()

		# #tokenize pred_graphs
		pred_graphs, ents, raw_ents = self.g2t_model.g2t_preprocess(pred_graphs)
		pred_graphs_ids = self.g2t_model.tokenizer(pred_graphs, return_tensors='pt', truncation=True, padding=True).input_ids
		gold_text_ids = self.g2t_model.tokenizer(gold_text, return_tensors='pt').input_ids
		loss = self.g2t_model(input_ids = pred_graphs_ids, labels = gold_text_ids).loss
		loss.backward()
		self.g2t_opt.step()

		# # CODE IS UNNECESSARY IF WE ARE BLACKBOXING LOSS FUNCTION - DAVID's CODE HERE

		# text_log_probs = self.g2t_model.t5_model.model.forward(pred_graphs) # bs x out_text_len x vocab_size need to check implementation of forward
		# _, out_text_len, vocab_size = text_log_probs.shape
		# #pred_text = self.g2t_model.predict(pred_graphs)   #note: this would not be predict here - it would be calling running through the model i think
		

		# if gold_text_len < out_text_len:
		# 	empties = torch.ones(bs, out_text_len - gold_text_len) * self.vocab.text.word2idx["<EMPTY>"] 
		# 	gold_text = torch.cat((gold_text, empties), dim = 1)
		# elif gold_text_len > out_text_len: # need to double check
		# 	empty_probs = torch.ones(bs, gold_text_len - out_text_len, vocab_size) * float('-inf')
		# 	empty_probs[:, :, self.vocab.text.word2idx["<EMPTY>"]] = 0
		# 	text_log_probs = torch.cat((text_log_probs, empty_probs), dim = 1)

		# assert gold_text.shape[1] == text_log_probs.shape[1], "Gold mismatches text log probs length in T cycle"
		# # convert pred_text to tensor of word indices
		# loss = F.nll_loss(text_log_probs.view(-1, text_log_probs.shape[-1]), gold_text.view(-1), ignore_index=0) # could be wrong, again
		# loss.backward()
		# #nn.utils.clip_grad_norm_(g2t_model.parameters(), config['clip'])
		# self.g2t_opt.step()
		return loss.item()



	def g_cycle(self, graph_batch): # optimizes t2g
		"""
			Input: graph_batch: list (length batch_size) of dicts with entities and relations

			Performs G2T then optimizes T2G by computing loss of generated graph and original (gold) graph
		"""
		self.g2t_model.eval()
		self.t2g_model.train()
		max_ents = max([len(graph["entities"]) for graph in graph_batch])
		gold_graphs = [dp.relation2Indices(self.vocab, graph, max_ents) for graph in graph_batch]
		gold_graphs = torch.LongTensor(gold_graphs).to(self.device) # bs x max_ents x max_ents - used for loss computation
		with torch.no_grad():
			pred_text = self.g2t_model.predict(graph_batch)

		self.t2g_opt.zero_grad()
		pred_text, pred_text_ents = self.t2g_model.t2g_preprocess(pred_text)

		graph_log_probs = self.t2g_model.model.forward(pred_text.to(self.device), pred_text_ents.to(self.device)) # bs x max_ents x max_ents x num_relations - log probs of each relation between all entities in each batch
		loss = F.nll_loss(graph_log_probs.view(-1, graph_log_probs.shape[-1]), gold_graphs.view(-1), ignore_index=self.vocab.relations.word2idx['<EMPTY>']) # ignore index should be 0
		loss.backward()
		#nn.utils.clip_grad_norm_(g2t_model.parameters(), config['clip'])
		self.t2g_opt.step()
		return loss.item()

	def back_translation(self, text_batch, graph_batch):
		g_loss = self.g_cycle(graph_batch)
		t_loss = self.t_cycle(text_batch)
		return g_loss, t_loss

	def train(self, epochs, batch_size, learning_rate, shuffle):

		for i in range(epochs):
			tcycle_dataloader, gcycle_dataloader = dp.create_cycle_dataloader(raw_json_file=self.vocab.raw_data, batch_size = batch_size, shuffle=shuffle)
			dataloader = list(zip(tcycle_dataloader, gcycle_dataloader))
			for index, (tbatch, gbatch) in tqdm.tqdm(enumerate(dataloader)):
				g_loss, t_loss = self.back_translation(tbatch, gbatch)
                    

    
    