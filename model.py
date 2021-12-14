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
from transformers import T5ForConditionalGeneration

import data_processing as dp
import json

import t2g
import g2t 


class CycleModel():
	def __init__(self, vocab, evaluate=False, device = "cpu"):
		if device == "gpu":
			self.device = torch.device('cuda:0')
		else:
			self.device = torch.device('cpu')

		self.t2g_model = t2g.T2GModel(vocab, self.device, 768)
		self.g2t_model = g2t.G2TModel(vocab)
		
		self.vocab = vocab
		self.evaluate = evaluate

		self.best_g2t_average = -1
		self.best_t2g_average = -1

		self.init_g2t_dev()
	
	def init_g2t_dev(self):
		if self.evaluate:
			f_dev = open('json_datasets/test.json', 'r')
		else:
			f_dev = open('json_datasets/dev.json', 'r')
		rawest_dev = json.load(f_dev)
		#raw_dev = raw_dev
		raw_dev = []
		self.raw_test = []
        
		# for item in rawest_dev:
		# 	if len(item['entities']) > 0:
		# 		raw_dev.append(item)
		for raw_json_sentence in rawest_dev:
			_, entity_inds = dp.concatTextEntities(self.vocab, raw_json_sentence)
			if len(entity_inds) > 0:
				raw_dev.append(raw_json_sentence)
		f_dev.close()
		#raw_dev = raw_dev
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
        
		with torch.no_grad():
			pred_text = self.g2t_model.predict(graph_batch, replace_ents=True)

		self.t2g_opt.zero_grad()
		
		pred_text, pred_text_ents = self.t2g_model.t2g_preprocess(pred_text)

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

	

	def train(self, epochs, batch_size,  shuffle, t2g_lr=1e-3, g2t_lr=1e-3):
		self.t2g_opt = torch.optim.Adam(self.t2g_model.model.parameters(), lr=t2g_lr)
		self.g2t_opt = torch.optim.Adam(self.g2t_model.t5_model.parameters(), lr=g2t_lr)

		tcycle_dataloader, gcycle_dataloader = dp.create_cycle_dataloader(raw_json_file=self.vocab.raw_data, batch_size = batch_size, shuffle=shuffle)
		for i in range(epochs):
			dataloader = list(zip(tcycle_dataloader, gcycle_dataloader))
			print("num iterations", len(dataloader))
			dataloader = dataloader # TODO: REVERT
			for index, (tbatch, gbatch) in tqdm.tqdm(enumerate(dataloader)):
				g_loss, t_loss = self.back_translation(tbatch, gbatch)
				print()
				print("G-cycle loss", g_loss)
				print("T-cycle loss", t_loss)
			self.evaluate_model()
		self.eval_best_model()


	def load_best_model(self):
		self.t2g_model.eval()
		self.g2t_model.eval()
		print("Evaluating best model")
		self.g2t_model.t5_model = T5ForConditionalGeneration.from_pretrained('g2t.bin', return_dict=True,config='t5-base-config.json')
		print("Loaded G2T model")
		self.t2g_model.model.load_state_dict(torch.load('t2g.pt'))
		print("Loaded T2G model")

	def eval_best_model(self):
		self.load_best_model()
		self.evaluate_model(download=False)

	def evaluate_model(self, download=True):
		self.t2g_model.eval()
		self.g2t_model.eval()
		print("evaluating")
		hyp = self.g2t_model.predict(self.dev_graphs, replace_ents=False)    
		hyp = dict(zip(range(len(self.dev_graphs)), [[x.lower()] for x in hyp]))
		# ref = dict(zip(range(len(dev_df)), [[dev_df['target_text'][i]] for i in range(len(dev_df))]))
		#print(self.ref[:num_graphs])
		ret = self.bleu.compute_score(self.ref, hyp)
		#print('BLEU INP {0:}'.format(len(hyp)))
		bleu = ret[0][3]
		meteor = self.meteor.compute_score(self.ref, hyp)[0]
		rouge = self.rouge.compute_score(self.ref, hyp)[0]
		cider = self.cider.compute_score(self.ref, hyp)[0]
		print('BLEU 4 {0:}'.format(bleu))
		print('METEOR {0:}'.format(meteor))
		print('ROUGE_L {0:}'.format(rouge))
		print('Cider {0:}'.format(cider))

		g2t_average = (bleu + meteor + rouge + cider ) / 4.0
		print("Overall G2T (Average): ", g2t_average)
		if g2t_average > self.best_g2t_average and download:
			self.best_g2t_average = g2t_average
			print("Saving G2T model")
			torch.save(self.g2t_model.t5_model.state_dict(), 'g2t.bin')


		micro, macro, true, pred = self.t2g_model.eval_t2g(self.raw_dev)

		# print(self.raw_dev)

		print("Micro F1 Score: ", micro)
		print()
		print("Macro F1 Score: ", macro)
		print()

		t2g_average = (micro + macro) / 2.0
		print("Overall T2G (Average): ", t2g_average)
		if t2g_average > self.best_t2g_average and download:
			self.best_t2g_average = t2g_average
			print("Saving T2G model")
			torch.save(self.t2g_model.model.state_dict(), 't2g.pt')
		
        

    
def main():
                    
	# Opening JSON file
	f = open('json_datasets/train.json', 'r')

	raw_train = json.load(f)

	vocab = dp.Vocabulary()
	vocab.parseText(raw_train)

	#for training
	cycle_model = CycleModel(vocab)
	cycle_model.train(epochs=15, batch_size = 16, shuffle = True)
	#cycle_model.train(epochs=15, batch_size = 32, shuffle = True, t2g_lr = 5.0e-5, g2t_lr = 2.0e-4)

	# for evaluation
	# cycle_model = CycleModel(vocab, evaluate=True)
	# cycle_model.eval_best_model()

if __name__ == "__main__":
	# stuff only to run when not called via 'import' here
	main()