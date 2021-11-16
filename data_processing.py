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

		self.init_vocab(self.text)
		self.init_vocab(self.entities)
		self.init_vocab(self.relations)
		
		self.relations.word2idx["<NO_RELATION>"] = len(self.relations.wordlist) # no relation token for relations vocab
		self.relations.wordlist.append("<NO_RELATION>")
		
	# initializes UNK, SOS, EOS, and EMPTY tokens
	def init_vocab(self, vocab_category):
		tokens = ["<EMPTY>", "<UNK>", "<SOS>", "<EOS>"]

		for token in tokens:
			vocab_category.word2idx[token] = len(vocab_category.wordlist)
			vocab_category.wordlist.append(token)
		# vocab_category.word2idx["<UNK>"] = len(vocab_category.wordlist)
		# vocab_category.wordlist.append("<UNK>")
		# vocab_category.word2idx["<SOS>"] = len(vocab_category.wordlist)
		# vocab_category.wordlist.append("<SOS>")
		# vocab_category.word2idx["<EOS>"] = len(vocab_category.wordlist)
		# vocab_category.wordlist.append("<EOS>")
		# vocab_category.word2idx["<EMPTY>"] = len(vocab_category.wordlist)
		# vocab_category.wordlist.append("<EMPTY>")
		

	def parseSentence(self, raw_json_sentence):
		for relation in raw_json_sentence['relations']: #Relation parsing here
			assert len(relation) == 3, "CHECK THIS!"
			if relation[1] not in self.relations.word2idx:
				self.relations.word2idx[relation[1]] = len(self.relations.wordlist)
				self.relations.wordlist.append(relation[1])
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
			self.entities.wordfreq += Counter(entity)
	
	def parseText(self, raw_json):
		for raw_sentence in raw_json:
			self.parseSentence(raw_sentence)
		print("Finished Parsing Text")

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

def relation2Indices(vocab, raw_json_sentence):
	'''
		Parameters:
			vocab - Vocabulary object that contains the vocab from a parsed json file
			raw_json_sentence - one element of array contained in raw json file

		Return:
			labels - Symmetrical [len(entities) x len(entities)] Longtensor where 
			         labels[i][j] denotes the relation between entities i and j
	'''
	l = len(raw_json_sentence['entities'])
	ret = torch.ones((l,l), dtype = torch.long)*vocab.relations.word2idx["<NO_RELATION>"]
	entitydict = {}
	for i, entity in enumerate(raw_json_sentence['entities']):
		entitydict["".join(entity)] = i
	for relation in raw_json_sentence['relations']:
		ind1 = entitydict["".join(relation[0])]
		ind2 = entitydict["".join(relation[2])]
		ret[ind1][ind2] = ret[ind2][ind1] = vocab.relations.word2idx[relation[1]]
	return ret

		

def concatTextEntities(vocab, raw_json_sentence, entity_indices):
	sent = text2Indices(vocab, raw_json_sentence['text'])
	modified_input = torch.LongTensor([0])
	lbound = 0
	entity_locations = []
	additional_words = 0
	for index, value in enumerate(sent):
		if value.item() in entity_indices:
			temp = entity2Indices(vocab, raw_json_sentence['entities'][entity_indices[value.item()]])
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
		
		self.entity_indices = getEntityIndices(self.vocab)

		for raw_json_sentence in raw_json_file:
			self.labels.append(relation2Indices(self.vocab, raw_json_sentence))
			self.inputs.append(concatTextEntities(self.vocab, raw_json_sentence, self.entity_indices))

		print("Finished processing raw json file")

	def __len__(self):
		return len(self.inputs)
	def __getitem__(self, idx):
		return self.inputs[idx], self.labels[idx]

def getBatches(vocab, dataset, batch_size, shuffle = False):
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

		final_text = torch.ones((len(indices), maxlentext), dtype = torch.long)*vocab.text.word2idx["<EMPTY>"]
		final_label = torch.ones((len(indices), maxlenentity, maxlenentity), dtype = torch.long)*vocab.relations.word2idx["<EMPTY>"]
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