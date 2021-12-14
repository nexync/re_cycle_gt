import t2g
import json
import data_processing as dp
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import f1_score

def train_model_supervised(model, dev_file, warmup_epochs = 3, learning_rate = 5.0e-5, epochs = 20):
	"""
	"""
	num_relations = len(model.vocab.relations.wordlist)

	# Create model
	optimzer = torch.optim.Adam(model.model.parameters(), lr = learning_rate)
	best_acc = 0
	state_dict_clone = {k: v.clone() for k, v in model.model.state_dict().items()}
	for p in range(epochs):
		model.model.train()
		if p < warmup_epochs:
			t, g = dp.create_cycle_dataloader(model.vocab.raw_data[:3550], 32, True)
			dataloader = t+g

		else:
			t, g = dp.create_cycle_dataloader(model.vocab.raw_data, 32, True)
			dataloader = t+g
		
		loss_this_epoch = 0.0
		for index, batch in tqdm.tqdm(enumerate(dataloader)):
			pre_text, pre_ents = model.t2g_preprocess(batch)

			bs, _ = pre_text.shape

			max_ents = max([len(ex['entities']) for ex in batch])

			labels = torch.zeros((bs, max_ents, max_ents), dtype = torch.long)
			for k, raw_json in enumerate(batch):
				labels[k] = dp.relation2Indices(vocab, raw_json, max_ents)
    
			log_probs = model.model(pre_text, pre_ents, torch.tensor(max_ents))

			loss = F.nll_loss(log_probs.view(-1, num_relations), labels.view(-1), ignore_index = 0)
			loss_this_epoch += loss.item()
			optimzer.zero_grad()
			loss.backward()
			# torch.nn.utils.clip_grad_norm_(
			#     [p for group in optimzer.param_groups for p in group['params']], CLIP)
			optimzer.step()

		print("Evaluating: ")
		micro, macro = model.eval_t2g(dev_file)
		if (micro + macro)/2 > best_acc:
			best_acc = (micro+macro)/2
			curr_state_dict = model.model.state_dict()
			for key in state_dict_clone.keys():
				state_dict_clone[key].copy_(curr_state_dict[key])
		

		print("Loss this epoch: ", loss_this_epoch/len(dataloader)*32)

	curr_state_dict = model.model.state_dict()
	for key in state_dict_clone.keys():
		curr_state_dict[key].copy_(state_dict_clone[key])

	print("saving model")
	return curr_state_dict

f = open('json_datasets/train.json', 'r')
raw_train = json.load(f)

f = open('json_datasets/dev.json', 'r')
raw_dev = json.load(f)

f = open('json_datasets/test.json', 'r')
raw_test = json.load(f)

vocab = dp.Vocabulary()
vocab.parseText(raw_train)

t2g_model = t2g.T2GModel(vocab, torch.device('cpu'), 512)

state_dict = train_model_supervised(t2g_model, raw_dev)
#save best model
torch.save(state_dict, "sup_t2g_dict")


#loading best model
t2g_model_infer = t2g.T2GModel(vocab, torch.device('cpu'), 512)
t2g_model_infer.model.load_state_dict(torch.load("sup_t2g_dict"))


#infer on test
tr = []
pre = []

for index in range(len(raw_test)):
	try:
		_,_, t, p = t2g_model_infer.eval_t2g(raw_test[index: index+1])
		tr.extend(t)
		pre.extend(p)
	except RuntimeError:
		continue

print("Micro F1: ", f1_score(tr, pre, average = "micro"))
print("Macro F1: ", f1_score(tr, pre, average = "macro"))


