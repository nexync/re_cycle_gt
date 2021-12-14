#Imports

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tqdm
from sklearn.metrics import f1_score

class ModelLSTM(nn.Module):
	def __init__(self, input_types, relation_types, model_dim, dropout = 0.0):
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

	#def forward(self, batch):
	def forward(self, sents, ent_inds, max_ents):
		#sents = batch['text']
		sents, (c_0, h_0) = self.lstm(self.emb(sents))

		bs, _, hidden_dim = sents.shape
		
		#max_ents = max([max([ent_ind[0] for ent_ind in batch_ent_inds]) for batch_ent_inds in ent_inds]).item() + 1
		max_ents = max_ents.item()

		cont_word_mask = sents.new_zeros(bs, max_ents)
		cont_word_embs = sents.new_zeros(bs, max_ents, hidden_dim)

		#for b, (sent,entind) in enumerate(zip(sents,batch['entity_inds'])):
		for b, (sent,entind) in enumerate(zip(sents, ent_inds)):
			for z in entind:
				if z[0] == -1:
					break
				else:
					wordemb = sent[z[1]:z[2]]
					#cont_word_embs[b, z[0]] = (cont_word_mask[b, z[0]]*cont_word_embs[b, z[0]] + torch.mean(wordemb, dim = 0))/(cont_word_mask[b, z[0]] + 1)
					#FUCK, try ignoring repeats for now :(
					cont_word_embs[b, z[0]] = torch.mean(wordemb, dim =0)
					cont_word_mask[b, z[0]] = 1

		# bs x max_ents x model_dim
		cont_word_embs = self.layer_norm(cont_word_embs)
		cont_word_mask = torch.clamp(cont_word_mask, 0, 1)

		rel1 = self.relation_layer1(cont_word_embs)
		rel2 = self.relation_layer2(cont_word_embs)

		#bs x max_ents x max_ents x model_dim
		out = rel1.unsqueeze(1) + rel2.unsqueeze(2)

		out = F.relu(self.drop(out))
		out = F.relu(self.projection(out))
		out = self.decoder(out)

		out = out * cont_word_mask.view(bs,max_ents,1,1) * cont_word_mask.view(bs,1,max_ents,1)

		return torch.log_softmax(out, -1) # bs x max ents x max_ents x num_relations

class T2GModel():
	def __init__(self, vocab, device, model_dim):
		self.inp_types = len(vocab.entities.wordlist) + len(vocab.text.wordlist)
		self.rel_types = len(vocab.relations.wordlist)

		self.model = ModelLSTM(self.inp_types, self.rel_types, model_dim = model_dim).to(device)
		self.vocab = vocab
		self.device = device

	def eval(self):
		self.model.eval()
    
	def train(self):
		self.model.train()
	def t2g_preprocess(self, batch):
			""" 
				input: list of dictionaries in raw_json_format
				output: prepreprocessed dictionaries containing text, entity inds
			"""

			def entity2Indices(entity, mode = "T2G"):
				temp = torch.zeros(len(entity), dtype = torch.long)
				for ind, word in enumerate(entity):
					if word not in self.vocab.entities.word2idx:
						temp[ind] = self.vocab.entities.word2idx["<UNK>"]
					else:
						temp[ind] = self.vocab.entities.word2idx[word]
				return temp
					
			def text2Indices(text):
				temp = torch.zeros(len(text.split()) + 2, dtype=torch.long)
				temp[0] = self.vocab.text.word2idx["<SOS>"]
				for ind, word in enumerate(text.split()):
					if word not in self.vocab.text.word2idx:
						temp[ind + 1] = self.vocab.text.word2idx["<UNK>"]
					else:
						temp[ind + 1] = self.vocab.text.word2idx[word]
				temp[-1] = self.vocab.text.word2idx["<EOS>"]
				return temp

			def concatTextEntities(raw_json_sentence):
				sent = text2Indices(raw_json_sentence['text'])
				modified_input = torch.LongTensor([0])
				lbound = 0
				entity_locations = []
				additional_words = 0
				for index, value in enumerate(sent):
					if value.item() in self.vocab.entityindices:
						temp = entity2Indices(raw_json_sentence['entities'][self.vocab.entityindices[value.item()]])
						temp += len(self.vocab.text.wordlist)
						modified_input = torch.cat((modified_input, sent[lbound:index], temp), dim = 0)
						entity_locations.append([self.vocab.entityindices[value.item()], index + additional_words, index + additional_words + len(temp)])
						additional_words += len(temp) - 1
						lbound = index + 1
				modified_input = torch.cat((modified_input, sent[lbound:]), dim = 0)[1:]
				return modified_input, torch.LongTensor(entity_locations)

			maxlentext = 0
			maxents = 0
			temp_text = []
			temp_inds = []
			skipped = 0
			for k, raw_json_sentence in enumerate(batch):
				
				(text, entity_inds) = concatTextEntities(raw_json_sentence)
				if len(entity_inds) == 0:
					skipped += 1
					continue
				temp_inds.append(entity_inds)
				if len(entity_inds) > maxents:
					maxents = len(entity_inds)
				temp_text.append(text)
				if text.shape[0] > maxlentext:
					maxlentext = text.shape[0]
				
			final_text = torch.ones((len(batch)-skipped, maxlentext), dtype = torch.long)*self.vocab.text.word2idx["<EMPTY>"]
			final_ents = torch.ones((len(batch)-skipped, maxents, 3), dtype = torch.long)*-1

			for k in range(len(batch)-skipped):
				# print(temp_text[k], temp_inds[k])
				# print(len(temp_text[k]), len(temp_inds[k]), k)
				final_text[k][:len(temp_text[k])] = temp_text[k]
				final_ents[k][:len(temp_inds[k])] = temp_inds[k]
			
			return final_text, final_ents

	# input - texts with original entities taken out (list of dicts with text and entities)
	# output - batch of graphs (list of dicts with relations and entities)
	def predict(self, batch):
		preprocessed_text, preprocessed_inds = self.t2g_preprocess(batch)
		max_ents = max([len(graph["entities"]) for graph in batch])

		#preds = self.model(preprocessed_text.to(self.device), preprocessed_inds.to(self.device))
		preds = self.model(preprocessed_text.to(self.device), preprocessed_inds.to(self.device), torch.tensor(max_ents).to(self.device))
		preds = torch.argmax(preds, -1)

		bs, ne, _ = preds.shape

		output = [] #list of dictionaries

		#print(ne, max([len(batch[b]['entities'])] for b in range(bs)))
		assert ne == max([len(batch[b]['entities'])] for b in range(bs))[0]

		for b in range(bs):
			temp = {
				"relations": [],
				"entities": batch[b]['entities']
			}
			for i in range(0, len(temp['entities'])):
				for j in range(i+1, len(temp['entities'])):
					temp['relations'].append([temp['entities'][i], self.vocab.relations.wordlist[preds[b, i, j]], temp['entities'][j]])
			output.append(temp)
		return output

	def eval_t2g(self, eval_dataset):
		def relation2Indices(raw_json_sentence, max_ents):
			'''
				Parameters:
					vocab - Vocabulary object that contains the vocab from a parsed json file
					raw_json_sentence - one element of array (i.e. one dict) contained in raw json file
					max_ents - gives size of return array

				Return:
					labels - Symmetrical [max_entities x max_entities)] Longtensor where 
									labels[i][j] denotes the relation between entities i and j.
							Anything where i >= l or j >= l is <EMPTY> 
			'''
			l = len(raw_json_sentence['entities'])
			ret = torch.ones((max_ents,max_ents), dtype = torch.long)*self.vocab.relations.word2idx["<NO_RELATION>"]
			for i in range(l, max_ents):
				for j in range(0, max_ents):
					ret[i][j] = ret[j][i] = self.vocab.relations.word2idx["<EMPTY>"]
			
			for i in range(0, l):
				for j in range(0, i+1):
					ret[i][j] = self.vocab.relations.word2idx["<EMPTY>"]
					
			# for i in range(l, max_ents):
			# 	for j in range(0, max_ents): # could do (0, l) for efficiency
			# 		ret[j][i] = vocab.relations.word2idx["<EMPTY>"]
			entitydict = {}
			for i, entity in enumerate(raw_json_sentence['entities']):
				entitydict["".join(entity)] = i
			for relation in raw_json_sentence['relations']:
				ind1 = entitydict["".join(relation[0])]
				ind2 = entitydict["".join(relation[2])]
				if relation[1] not in self.vocab.relations.word2idx:
					rep = self.vocab.relations.word2idx['<UNK>']
				else:
					rep = self.vocab.relations.word2idx[relation[1]]
				#ret[ind1][ind2] = ret[ind2][ind1] = vocab.relations.word2idx[relation[1]]
				if ind1 < ind2:
					ret[ind1][ind2] = rep
					#self.vocab.relations.word2idx[relation[1]]
					#ret[ind2][ind1] = self.vocab.relations.word2idx["<EMPTY>"]
				else:
					ret[ind2][ind1] = rep
					#self.vocab.relations.word2idx[relation[1]]
					#ret[ind1][ind2] = self.vocab.relations.word2idx["<EMPTY>"]
			return ret

		self.model.eval()
		preprocessed_text, preprocessed_inds = self.t2g_preprocess(eval_dataset)
		max_ents = max([len(graph["entities"]) for graph in eval_dataset])

		preprocessed_labels = [relation2Indices(json_sent, max_ents) for json_sent in eval_dataset]

		
		preds = self.model(preprocessed_text.to(self.device), preprocessed_inds.to(self.device), torch.tensor(max_ents))
		preds = torch.argmax(preds, -1)

		bs, ne, _ = preds.shape

		true_labels = []
		pred_labels = []

		for b in range(bs):
			temp_true = []
			temp_pred = []
			for i in range(0, len(eval_dataset[b]['entities'])):
				for j in range(i+1, len(eval_dataset[b]['entities'])):
					temp_true.append(preprocessed_labels[b][i][j].item())
					temp_pred.append(preds[b][i][j].item())
			true_labels.extend(temp_true)
			pred_labels.extend(temp_pred)

		# print("Micro F1: ", f1_score(true_labels, pred_labels, average = 'micro'))
		# print("Macro F1: ", f1_score(true_labels, pred_labels, average = 'macro'))
		#print("true", true_labels)
		#print("pred", pred_labels)
		return f1_score(true_labels, pred_labels, average = 'micro'), f1_score(true_labels, pred_labels, average = 'macro'), true_labels, pred_labels