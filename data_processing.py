#Imports

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import tqdm
import json
import random
import re

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class VocabCategory():
	def __init__(self):
		self.wordlist = []
		self.word2idx = {}
		self.wordfreq = Counter()

class Vocabulary():
	def __init__(self):
		print("Creating empty vocabulary object")
		self.text = VocabCategory()
		self.entities = VocabCategory()
		self.relations = VocabCategory()

		self.raw_data = []
		self.entityindices = {}

		self.init_vocab(self.text)
		self.init_vocab(self.entities)
		self.init_vocab(self.relations)
		
		self.relations.word2idx["<NO_RELATION>"] = len(self.relations.wordlist) # no relation token for relations vocab
		self.relations.wordlist.append("<NO_RELATION>")

		relations = ["<H>", "<R>", "<T>", "g2t:"]

		for token in relations:
			self.text.word2idx[token] = len(self.text.wordlist)
			self.text.wordlist.append(token)
		
	# initializes UNK, SOS, EOS, and EMPTY tokens
	def init_vocab(self, vocab_category):
		tokens = ["<EMPTY>", "<UNK>", "<SOS>", "<EOS>"]

		for token in tokens:
			vocab_category.word2idx[token] = len(vocab_category.wordlist)
			vocab_category.wordlist.append(token)


	def parseSentence(self, raw_json_sentence):
		for relation in raw_json_sentence['relations']: #Relation parsing here
			assert len(relation) == 3, "CHECK THIS!"
			if relation[1] not in self.relations.word2idx:
				self.relations.word2idx[relation[1]] = len(self.relations.wordlist)
				self.relations.wordlist.append(relation[1])
				for word in camelCaseSplit(relation[1]):
					if word not in self.text.word2idx:
						self.text.word2idx[word] = len(self.text.wordlist)
						self.text.wordlist.append(word)
			self.relations.wordfreq.update({relation[1]: 1})
		
		for word in raw_json_sentence['text'].split(): #Word parsing here
			if word not in self.text.word2idx:
				self.text.word2idx[word] = len(self.text.wordlist)
				self.text.wordlist.append(word)
		self.text.wordfreq += Counter(raw_json_sentence['text'].split())

		for entity in raw_json_sentence['entities']:
			for e in entity:
				if e not in self.entities.word2idx:
					self.entities.word2idx[e] = len(self.entities.wordlist)
					self.entities.wordlist.append(e)
				if e not in self.text.word2idx:
					self.text.word2idx[e] = len(self.text.wordlist)
					self.text.wordlist.append(e)
			self.entities.wordfreq += Counter(entity)
	
	def parseText(self, raw_json):
		for raw_sentence in raw_json:
			self.parseSentence(raw_sentence)
		self.raw_data += raw_json
		self.entityindices = self.getEntityIndices()
		print("Finished Parsing Text")

	def getEntityIndices(self):
		entity_indices = {}
		i = 0
		while True:
			if '<ENT_' + str(i) + '>' in self.text.word2idx:
				entity_indices[(self.text.word2idx['<ENT_' + str(i) + '>'])] = i
				i += 1
			else:
				return entity_indices

def entity2Indices(vocab, entity):
	'''
		Parameters:
			vocab
			entity - an entity given as a string array
		Return:
			indices - a len(entity) LongTensor whose values are the indices of words
	'''
	temp = torch.zeros(len(entity), dtype = torch.long)
	for ind, word in enumerate(entity):
		if word not in vocab.entities.word2idx:
			temp[ind] = vocab.entities.word2idx["<UNK>"]
		else:
			temp[ind] = vocab.entities.word2idx[word]
	return temp

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
		
def text2Indices(vocab, text):
	'''
		Parameters:
			vocab
			text - a string of text
		Return:
			indices - a len(text.split() + 2) LongTensor whose values are the indices of words (starts with SOS and EOS)
	'''
	temp = torch.zeros(len(text.split()) + 2, dtype=torch.long)
	temp[0] = vocab.text.word2idx["<SOS>"]
	for ind, word in enumerate(text.split()):
		if word not in vocab.text.word2idx:
			temp[ind + 1] = vocab.text.word2idx["<UNK>"]
		else:
			temp[ind + 1] = vocab.text.word2idx[word]
	temp[-1] = vocab.text.word2idx["<EOS>"]
	return temp

def relation2Indices(vocab, raw_json_sentence, max_ents):
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
	ret = torch.ones((max_ents,max_ents), dtype = torch.long)*vocab.relations.word2idx["<NO_RELATION>"]
	for i in range(l, max_ents):
		for j in range(0, max_ents):
			ret[i][j] = ret[j][i] =  vocab.relations.word2idx["<EMPTY>"]
	
	for i in range(0, l):
		for j in range(0, i+1):
			ret[i][j] = vocab.relations.word2idx["<EMPTY>"]
			
	# for i in range(l, max_ents):
	# 	for j in range(0, max_ents): # could do (0, l) for efficiency
	# 		ret[j][i] = vocab.relations.word2idx["<EMPTY>"]
	entitydict = {}
	for i, entity in enumerate(raw_json_sentence['entities']):
		entitydict["".join(entity)] = i
	for relation in raw_json_sentence['relations']:
		ind1 = entitydict["".join(relation[0])]
		ind2 = entitydict["".join(relation[2])]
		#ret[ind1][ind2] = ret[ind2][ind1] = vocab.relations.word2idx[relation[1]]
		if relation[1] not in vocab.relations.word2idx:
			ret[ind1][ind2] = vocab.relations.word2idx["<UNK>"]
		else:
			if ind1 < ind2:
				ret[ind1][ind2] = vocab.relations.word2idx[relation[1]]
				#ret[ind2][ind1] = vocab.relations.word2idx["<EMPTY>"]
			else:
				ret[ind2][ind1] = vocab.relations.word2idx[relation[1]]
				#ret[ind1][ind2] = vocab.relations.word2idx["<EMPTY>"]
	return ret


def concatTextEntities(vocab, raw_json_sentence):
	sent = text2Indices(vocab, raw_json_sentence['text'])
	modified_input = torch.LongTensor([0])
	lbound = 0
	entity_locations = []
	additional_words = 0
	for index, value in enumerate(sent):
		if value.item() in vocab.entityindices:
			temp = entity2Indices(vocab, raw_json_sentence['entities'][vocab.entityindices[value.item()]])
			temp += len(vocab.text.wordlist)
			modified_input = torch.cat((modified_input, sent[lbound:index], temp), dim = 0)
			entity_locations.append((index + additional_words, index + additional_words + len(temp)))
			additional_words += len(temp) - 1
			lbound = index + 1
			
	modified_input = torch.cat((modified_input, sent[lbound:]), dim = 0)[1:]

	return modified_input, entity_locations

def getEntityIndices(vocab):
	entity_indices = {}
	i = 0
	while True:
		if '<ENT_' + str(i) + '>' in vocab.text.word2idx:
			entity_indices[(vocab.text.word2idx['<ENT_' + str(i) + '>'])] = i
			i += 1
		else:
			return entity_indices

class text2GraphDataset(Dataset):
	def __init__(self, raw_json_file):
		print("Creating custom dataset for T2G task")
		
		self.vocab = Vocabulary()
		self.vocab.parseText(raw_json_file)
		
		self.inputs = []
		self.labels = []

		for raw_json_sentence in raw_json_file:
			self.labels.append(relation2Indices(self.vocab, raw_json_sentence, len(raw_json_sentence['entities'])))
			self.inputs.append(concatTextEntities(self.vocab, raw_json_sentence))

		print("Finished processing raw json file")

	def __len__(self):
		return len(self.inputs)
	def __getitem__(self, idx):
		return self.inputs[idx], self.labels[idx]

def getBatches(dataset, batch_size, shuffle = False):
	def create_dict(dataset, indices):
		tempdict = {
			"entity_inds": [],
			"text_lengths": [],
			"entity_lengths": []
		}
		
		for index in indices:
			(text, entity), label = dataset[index]
			tempdict["entity_inds"].append(entity)
			tempdict["text_lengths"].append(text.shape[0])
			tempdict["entity_lengths"].append(label.shape[0])
			#tempdict["text"].append(text)
			#tempdict["labels"].append(label)	
		maxlentext = max(tempdict["text_lengths"])
		maxlenentity = max(tempdict["entity_lengths"])

		final_text = torch.ones((len(indices), maxlentext), dtype = torch.long)*dataset.vocab.text.word2idx["<EMPTY>"]
		final_label = torch.ones((len(indices), maxlenentity, maxlenentity), dtype = torch.long)*dataset.vocab.relations.word2idx["<EMPTY>"]
		for k, index in enumerate(indices):
			(text, entity), label = dataset[index]
			final_text[k][:text.shape[0]] = text
			final_label[k][:label.shape[0],:label.shape[1]] = label

		tempdict["text"] = final_text
		tempdict["labels"] = final_label
		return tempdict

	indices = np.arange(0, len(dataset))
	if shuffle:
		random.shuffle(indices)
	
	assert len(indices) == len(dataset), "Check length"
	batches = []
	currIndex = 0
	while currIndex + batch_size <= len(dataset):
		tempdict = create_dict(dataset, indices[currIndex: currIndex + batch_size])
		batches.append(tempdict)
		currIndex += batch_size

	if currIndex < len(dataset):
		tempdict = create_dict(dataset, indices[currIndex:])
		batches.append(tempdict)
	return batches

def create_cycle_dataloader(raw_json_file, batch_size, shuffle = False):
	arr = np.array(raw_json_file)
	indices = np.arange(0, len(arr))
	
	if shuffle:
		random.shuffle(indices)
	
	t_indices = indices[:len(indices)//2]
	g_indices = indices[len(indices)//2:]

	tcycle = []
	gcycle = []

	currIndex = 0
	while currIndex + batch_size <= min(len(t_indices), len(g_indices)):
		tcycle.append(arr[t_indices[currIndex: currIndex + batch_size]])
		gcycle.append(arr[g_indices[currIndex: currIndex + batch_size]])
		currIndex += batch_size

	if currIndex < len(t_indices):
		tcycle.append(arr[t_indices[currIndex:]])
	if currIndex < len(g_indices):
		gcycle.append(arr[g_indices[currIndex:]])

	return tcycle, gcycle