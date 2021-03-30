import pandas as pd
import torch
from torch.utils.data import Dataset,RandomSampler,DataLoader
import numpy as np
import re
import ahocorasick

class CustomDataset(Dataset):
	def __init__(self,text,group_feat,targets,tokenizer,max_len):
		self.text = text
		self.group_feat = group_feat
		self.targets = targets
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self):
		return len(self.text)

	def __getitem__(self, item):
		text = self.text[item]
		target = self.targets[item]
		group_feat = self.group_feat[item]
		encoding = self.tokenizer(
			text,
			truncation=True,
			max_length=self.max_len,
			padding='max_length',
			return_tensors='pt',
		)
		return {
		  'text': text,
		  'group_feat':group_feat,
		  'input_ids': encoding['input_ids'].flatten(),
		  'attention_mask': encoding['attention_mask'].flatten(),
		  'targets': torch.tensor(target, dtype=torch.long),
		}

def get_data_loader(path,group_model,tokenizer,max_len,batch_size):
	# data is stored with its context, in case we want to train a model using the context as well
	dataset = pd.read_csv(path, sep='\t', names = ['targets', 'target_names', 'text', 'origin', 'main_text', 'secondary_text', 'source'])
	dataset = remove_invalid_inputs(dataset,'text')

	group_model_p,groups,group_tree = process_group_model(group_model)
	group_features = get_group_features(dataset.main_text.tolist(),group_model_p)

	data = CustomDataset(
					text=dataset.text.to_numpy(),
					group_feat=group_features,
					targets=dataset.targets.to_numpy(),
					tokenizer=tokenizer,
					max_len=max_len
					)

	sampler = RandomSampler(data)
	dataloader = DataLoader(data,batch_size=batch_size,sampler=sampler,pin_memory=True)
	return dataloader,data

def remove_invalid_inputs(dataset,text_column):
	'Simpel metode til at fjerne alle rækker fra en dataframe, baseret på om værdierne i en kolonne er af typen str'
	dataset['valid'] = dataset[text_column].apply(lambda x: isinstance(x, str))
	return dataset.loc[dataset.valid]

def get_group_features(texts,group_model):
	topic_automaton = prepare_aho_corasick_search(group_model)
	features = []
	for text in texts:
		t = torch.zeros(1,20,dtype=torch.int)
		for _,tup in topic_automaton.iter(str(text)):
			t[:,tup[0]] = 1
		features.append(t)
	return features

def process_group_model(topic_df):
	'Metode til at behandle en dataframe med den rå csv fil man downloader fra gsheets'
	# create and build the topic_model
	topic_model = {}
	topics = {}
	topic_tree = []
	remove_pattern = '[^a-zA-ZÆØÅæøå_ ]'
	c = 0
	for i,col in topic_df.iteritems():
		# first three rows in the file contains the topics
		# here we just normalize them
		l1 = re.sub(remove_pattern,'', i).lower()
		l2 = re.sub(remove_pattern,'', col[0]).lower()
		items = [l1,l2,c]
		c += 1

		topic_tree.append(items) 

		if l1 not in topics:
			topics[l1] ='level_1'
		if l2 not in topics:
			topics[l2] ='level_2'

		# build the topic_model        
		for item in items:
			if item not in topic_model:
				topic_model[item] = []
						
		# add tokens to the topic_model
		for token in col[1:]:
			if token == token:
				for item in items:
					topic_model[item].append(token)

	topic_tree = pd.DataFrame(topic_tree, columns=['group_lvl1', 'group_lvl2','group_id'])
	return topic_model,topics, topic_tree

def prepare_aho_corasick_search(topics_dict):
	a = ahocorasick.Automaton()
	for key,terms in topics_dict.items():
		for term in terms:
			a.add_word(term.lower().replace('_',' '), (key, term))
	a.make_automaton()
	return a
