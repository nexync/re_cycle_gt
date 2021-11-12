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
	def parseSentence(self, raw_json_sentence):
		for relation in raw_json_sentence['relations']: #Relation parsing here
			assert len(relation) == 3, "CHECK THIS!"
			if relation[1] not in self.relations.word2idx:
				self.relations.word2idx[relation[1]] = len(self.relations.wordlist)+1
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

def text2Relation(vocab, raw_json_sentence):
	'''
		Parameters:
			vocab - Vocabulary object that contains the vocab from a parsed json file
			raw_json_sentence - one element of array contained in raw json file

		Return:
			labels - Symmetrical [len(entities) x len(entities)] Longtensor where 
			         labels[i][j] denotes the relation between entities i and j
	'''
	l = len(raw_json_sentence['entities'])
	labels = torch.zeros((l,l), dtype = torch.long)
	entitydict = {}
	for i, entity in enumerate(raw_json_sentence['entities']):
		entitydict["".join(entity)] = i
	print(entitydict)
	for relation in raw_json_sentence['relations']:
		print(relation)
		ind1 = entitydict["".join(relation[0])]
		ind2 = entitydict["".join(relation[2])]
		labels[ind1][ind2] = labels[ind2][ind1] = vocab.relations.word2idx[relation[1]]
	return labels

def entity2Indices(vocab, entity):
	'''
		Parameters:
			vocab
			entity - an entity given as a string array
		Return:
			indices - a len(entity) LongTensor whose values are the indices of words
	'''
	indices = torch.zeros(len(entity), dtype = torch.long)
	for ind, word in enumerate(entity):
		if word not in vocab.entities.word2idx:
			indices[ind] = -1
		else:
			indices[ind] = vocab.entities.word2idx[word]
	return indices
		
def text2Indices(vocab, text):
	'''
		Parameters:
			vocab
			text - a string of text
		Return:
			indices - a len(text.split()) LongTensor whose values are the indices of words
	'''
	temp = torch.zeros(len(text.split()), dtype=torch.long)
	for ind, word in enumerate(text.split()):
		if word not in vocab.text.word2idx:
			temp[ind] = -1
		else:
			temp[ind] = vocab.text.word2idx[word]
	return temp
		

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