#Imports

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class ModelLSTM(nn.Module):
	def __init__(self, input_types, relation_types, model_dim, dropout = 0.5):
		super().__init__()

		self.word_types = input_types
		self.relation_types = relation_types
		self.dropout = dropout
		self.model_dim = model_dim

		self.emb = nn.Embedding(input_types, self.model_dim)
		self.lstm = nn.LSTM(self.model_dim, self.model_dim//2, batch_first=True, bidirectional=True, num_layers=2)
		self.relation_layer1 = nn.Linear(self.model_dim , self.model_dim)
		self.relation_layer2 = nn.Linear(self.model_dim , self.model_dim)
		self.drop = nn.Dropout(self.dropout)
		self.projection = nn.Linear(self.model_dim , self.model_dim)
		self.decoder = nn.Linear(self.model_dim , self.relation_types)
		self.layer_norm = nn.LayerNorm(self.model_dim)

		self.init_params()

	def init_params(self):
		nn.init.xavier_normal_(self.relation_layer1.weight.data)
		nn.init.xavier_normal_(self.relation_layer2.weight.data)
		nn.init.xavier_normal_(self.projection.weight.data)
		nn.init.xavier_normal_(self.decoder.weight.data)

		nn.init.constant_(self.relation_layer1.bias.data , 0)
		nn.init.constant_(self.relation_layer2.bias.data , 0)
		nn.init.constant_(self.projection.bias.data , 0)
		nn.init.constant_(self.decoder.bias.data , 0)

	def forward(self, batch):
		sents = batch['text']
		sents, (c_0, h_0) = self.lstm(self.emb(sents))

		bs, _, hidden_dim = sents.shape
		max_ents = max([len(x) for x in batch['entity_inds']])
		
		cont_word_mask = sents.new_zeros(bs, max_ents)
		cont_word_embs = sents.new_zeros(bs, max_ents, hidden_dim)

		for b, (sent,entind) in enumerate(zip(sents,batch['entity_inds'])):
			for n_ent, wordemb in enumerate([sent[z[0]:z[1]] for z in entind]):
				cont_word_embs[b, n_ent] = torch.mean(wordemb, dim = 0)
				cont_word_mask[b, n_ent] = 1

		# bs x max_ents x model_dim
		cont_word_embs = self.layer_norm(cont_word_embs)

		rel1 = self.relation_layer1(cont_word_embs)
		rel2 = self.relation_layer2(cont_word_embs)

		#bs x max_ents x max_ents x model_dim
		out = rel1.unsqueeze(1) + rel2.unsqueeze(2)

		out = self.drop(out)
		out = self.projection(out)
		out = self.decoder(out)

		out = out * cont_word_mask.view(bs,max_ents,1,1) * cont_word_mask.view(bs,1,max_ents,1)

		return torch.log_softmax(out, -1)