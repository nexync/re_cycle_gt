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
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

import data_processing as dp
import json

import t2g
import g2t 

# # instantiate models
# g2t_model = SimpleT5()
# t2g_model = t2g.T2GModel()

# # load (supports t5, mt5, byT5 models)
# g2t_model.from_pretrained("t5","t5-base")





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

		self.init_g2t_dev()
	
	def init_g2t_dev(self):

		f_dev = open('json_datasets/dev.json', 'r')
		raw_dev = json.load(f_dev)
		f_dev.close()
		raw_dev = raw_dev[0:100]
		self.dev_text, self.dev_graphs = [], []
		self.raw_dev = raw_dev
        
		graphs, entities, _ = self.g2t_model.g2t_preprocess(raw_dev, mode='G2T')
		for i, item in enumerate(raw_dev):
			ents = entities[i]
			text = item['text']
			for i in range(len(ents)):
				text = text.replace('<ENT_'+str(i)+'>', ents[i])
			graph = {}
			graph['entities'] = item['entities']
			graph['relations'] = item['relations']
			self.dev_graphs.append(graph)
			self.dev_text.append(text)
			
		self.ref = defaultdict(list)
		ptr = 0
		same = defaultdict(list)

		for i in range(len(self.dev_text)):
			if i > 0 and graphs[i] != graphs[i-1]:
				ptr += 1
			same[ptr].append(self.dev_text[i].lower())
			self.ref[i] = same[ptr]
		self.bleu = Bleu(4)
		self.meteor = Meteor()
		self.rouge = Rouge()
		self.cider = Cider()

	

    
	def t_cycle(self, text_batch): # optimizes g2t
		self.t2g_model.eval()
		self.g2t_model.train()

		gold_text = self.g2t_model.g2t_preprocess(text_batch, mode="TGT")
		#print(gold_text)
		# gold_text = gold_text.to(self.device) # bs x gold_text_len
		# bs, gold_text_len = gold_text.shape

		with torch.no_grad():
			pred_graphs = self.t2g_model.predict(text_batch) #synthetic batch of graphs
		
		self.g2t_opt.zero_grad()

		# #tokenize pred_graphs
		pred_graphs, ents, raw_ents = self.g2t_model.g2t_preprocess(pred_graphs, mode="G2T")
		pred_graphs_ids = self.g2t_model.tokenizer(pred_graphs, return_tensors='pt', truncation=True, padding=True).input_ids
		gold_text_ids = self.g2t_model.tokenizer(gold_text, return_tensors='pt', truncation=True, padding=True).input_ids
		loss = self.g2t_model.t5_model(input_ids = pred_graphs_ids, labels = gold_text_ids).loss
		loss.backward()
		self.g2t_opt.step()
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
		gold_graphs = torch.stack(gold_graphs)
		gold_graphs = gold_graphs.to(self.device) # bs x max_ents x max_ents - used for loss computation
        
		# print("gold")
		# print(gold_graphs)
		# print(gold_graphs.shape)
		# print()
        
		with torch.no_grad():
			pred_text = self.g2t_model.predict(graph_batch, replace_ents=True)
		#print(gold_graphs[0])
		#print(pred_text[0])
        
		# print("pred_text")
		# print(pred_text)
		# print(len(pred_text))
		# print()

		self.t2g_opt.zero_grad()
		
		pred_text, pred_text_ents = self.t2g_model.t2g_preprocess(pred_text)
        
		# print("pred_text processed")
		# print(pred_text)
		# print(pred_text.shape)
		# print()

		#graph_log_probs = self.t2g_model.model.forward(pred_text.to(self.device), pred_text_ents.to(self.device)) # bs x max_ents x max_ents x num_relations - log probs of each relation between all entities in each batch
		graph_log_probs = self.t2g_model.model.forward(pred_text.to(self.device), pred_text_ents.to(self.device), torch.tensor(max_ents).to(self.device)) # bs x max_ents x max_ents x num_relations - log probs of each relation between all entities in each batch

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
		tcycle_dataloader, gcycle_dataloader = dp.create_cycle_dataloader(raw_json_file=self.vocab.raw_data, batch_size = batch_size, shuffle=shuffle)
		for i in range(epochs):
			dataloader = list(zip(tcycle_dataloader, gcycle_dataloader))
			print("num iterations", len(dataloader))
			for index, (tbatch, gbatch) in tqdm.tqdm(enumerate(dataloader)):
				g_loss, t_loss = self.back_translation(tbatch, gbatch)
				print()
				print("G-cycle loss", g_loss)
				print("T-cycle loss", t_loss)
				if index % 100 == 1:
					self.evaluate_model()


	def evaluate_model(self):
		print("evaluating")
		hyp = self.g2t_model.predict(self.dev_graphs, replace_ents=False)    
		print("input graphs", self.dev_graphs)
		print()
		print("gold text", self.dev_text)
		print()
		print("hypothesized text", hyp)
		print()
		hyp = dict(zip(range(len(self.dev_graphs)), [[x.lower()] for x in hyp]))
		# ref = dict(zip(range(len(dev_df)), [[dev_df['target_text'][i]] for i in range(len(dev_df))]))
		#print(self.ref[:num_graphs])
		ret = self.bleu.compute_score(self.ref, hyp)
		print('BLEU INP {0:}'.format(len(hyp)))
		print('BLEU 1-4 {0:}'.format(ret[0]))
		print('METEOR {0:}'.format(self.meteor.compute_score(self.ref, hyp)[0]))
		print('ROUGE_L {0:}'.format(self.rouge.compute_score(self.ref, hyp)[0]))
		print('Cider {0:}'.format(self.cider.compute_score(self.ref, hyp)[0]))

		self.t2g_model.eval_t2g(self.raw_dev)
                    
# Opening JSON file
f = open('json_datasets/train.json', 'r')

raw_train = json.load(f)

vocab = dp.Vocabulary()
vocab.parseText(raw_train)

#create cycle

cycle_model = CycleModel(vocab)

cycle_model.evaluate_model()

#cycle_model.train(epochs=1, batch_size = 8, learning_rate = 0.1, shuffle = False)
    
    