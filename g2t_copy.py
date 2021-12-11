#Imports

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
# from simplet5 import SimpleT5
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

class G2TModel():
	def __init__(self, vocab):
		# instantiate
		# self.t5_model = SimpleT5()

		# # load (supports t5, mt5, byT5 models)
		# self.t5_model.from_pretrained("t5","t5-base")
		self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
		self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
		self.vocab = vocab

	def g2t_preprocess(self, raw):
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

		df = []
		graphs = []
		entities = []
		raw_ents = []
		for item in raw:
			graph = 'g2t:'
			for relation in item['relations']:
				graph += ' <H> ' + ' '.join(relation[0]) + ' <R> '
				relationName = ' '.join(camelCaseSplit(relation[1]))
				graph += relationName + ' <T> '
				graph += ' '.join(relation[2])

			ents = [' '.join(entity) for entity in item['entities']]
			graphs.append(graph)
			entities.append(ents)
			raw_ents.append(item['entities'])
		return graphs, entities, raw_ents

	def eval(self):
		self.t5_model.model.eval()
	
	def train(self):
		self.t5_model.model.train()

	# input: batch of graphs (list of dicts with relations and entities)
	# output: predicted texts with original entities taken out (list of dicts with text and entities)
	def predict(self, batch):
		def single_g2t(graph, ents, raw_ents):
			# predText = self.t5_model.predict(graph)
			graph_ids = self.tokenizer(graph, return_tensors='pt').input_ids
			output = self.t5_model.generate(graph_ids)
			predText = self.tokenizer.decode(output[0], skip_special_tokens=True)
			for i in range(len(ents)):
				if ents[i] in predText:
					predText.replace(ents[i], "<ENT_" + str(i) + ">")
				else:
					print("WARNING: ENTITY " + ents[i] + " NOT FOUND IN PREDICTED TEXT")
			return {'text' : predText, 'entities' : raw_ents}

		pGraphs, ents, raw_ents = self.g2t_preprocess(batch) # processed graphs, entities
		hyps = [single_g2t(pGraphs[i], ents[i], raw_ents[i]) for i in range(len(pGraphs))]

		return hyps


