import pandas as pd
import sys
from tqdm import tqdm
import torch
import transformers as tf
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import math
import numpy as np
import pickle
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime



def get_project_abstracts_labels(project):
	project_file = "full_exports_level1_level2_labels/" + str(project) + "/labels_" + str(project) + ".csv"
	dataset = pd.read_csv(project_file, encoding="ISO-8859-1", converters={'level2_labels':str})

	abstracts = []
	for abstract in tqdm(dataset['abstract']):
		abstracts.append(preprocess_abs(abstract))

	# print(abstracts)
	level1_labels = []
	for label in tqdm(dataset['level1_labels']):
		if label == 0 or label == 1:
			level1_labels.append(1)
		else:
			level1_labels.append(0)

	level2_labels = []
	for label in tqdm(dataset['level2_labels']):
		if label == "NA":
			level2_labels.append(0)
		else:
			level2_labels.append(1)

	return abstracts, level1_labels, level2_labels


def preprocess_abs(abstract):
	abstract = re.sub('[^A-Za-z]', ' ', str(abstract))
	abstract = abstract.lower()

	tokenized_abstract = word_tokenize(abstract)
	for word in tokenized_abstract:
		if word in stopwords.words("english"):
			tokenized_abstract.remove(word)

	processed_abstract = " ".join(tokenized_abstract)

	return processed_abstract

def bert_tokenize(abstracts, level1_labels, level2_labels):
	tokenizer = tf.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	input_ids = []
	attention_masks = []
	for abstract in tqdm(abstracts):
		encoded_dict = tokenizer.encode_plus(
						abstract,
						add_special_tokens = True,
						max_length = 256,
						pad_to_max_length = True,
						return_attention_mask = True,
						return_tensors = 'pt'
					)
		input_ids.append(encoded_dict['input_ids'])
		attention_masks.append(encoded_dict['attention_mask'])

	input_ids = torch.cat(input_ids, dim = 0)
	# print(input_ids)
	# print(input_ids.size())
	attention_masks = torch.cat(attention_masks, dim = 0)
	# print(attention_masks)
	# print(attention_masks.size())
	level1_labels = torch.tensor(level1_labels)
	# print(level1_labels)
	# print(level1_labels.size())
	level2_labels = torch.tensor(level2_labels)

	return input_ids, attention_masks, level1_labels, level2_labels

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def undersample(train_input_ids, train_attention_masks, train_level1_labels, train_level2_labels):
	# change all tensors to numpy arrays

	train_input_ids = train_input_ids.cpu().numpy()
	train_attention_masks = train_attention_masks.cpu().numpy()
	train_level1_labels = train_level1_labels.cpu().numpy()
	train_level2_labels = train_level2_labels.cpu().numpy()

	# undersampling
	np.random.seed(42)
	level1_pos_count = np.sum(train_level1_labels == 1)
	under_indexes = np.where(train_level1_labels == 1)[0]
	over_indexes = np.random.choice(np.where(train_level1_labels == 0)[0], level1_pos_count)
	train_input_ids = np.concatenate((train_input_ids[under_indexes], train_input_ids[over_indexes]), axis = 0)
	train_attention_masks = np.concatenate((train_attention_masks[under_indexes], train_attention_masks[over_indexes]), axis = 0)
	train_level1_labels = np.concatenate((train_level1_labels[under_indexes], train_level1_labels[over_indexes]), axis = 0)
	train_level2_labels = np.concatenate((train_level2_labels[under_indexes], train_level2_labels[over_indexes]), axis = 0)

	# change numpy arrays back to tensors
	train_input_ids = torch.tensor(train_input_ids)
	train_attention_masks = torch.tensor(train_attention_masks)
	train_level1_labels = torch.tensor(train_level1_labels)
	train_level2_labels = torch.tensor(train_level2_labels)

	'''
	print(train_input_ids.size())
	print(train_attention_masks.size())
	print(train_level1_labels.size())
	print(train_level2_labels.size())
	'''

	return train_input_ids, train_attention_masks, train_level1_labels, train_level2_labels


def train_bert(project, input_ids, attention_masks, level1_labels, level2_labels):
	train_indices = pickle.load(open("full_exports_level1_level2_labels/" + str(project) + "/train_indices.pkl", "rb"))
	dev_indices = pickle.load(open("full_exports_level1_level2_labels/" + str(project) + "/dev_indices.pkl", "rb"))
	test_indices = pickle.load(open("full_exports_level1_level2_labels/" + str(project) + "/test_indices.pkl", "rb"))

	train_input_ids = input_ids[torch.tensor(train_indices)]
	dev_input_ids = input_ids[torch.tensor(dev_indices)]
	test_input_ids = input_ids[torch.tensor(test_indices)]

	train_attention_masks = attention_masks[torch.tensor(train_indices)]
	dev_attention_masks = attention_masks[torch.tensor(dev_indices)]
	test_attention_masks = attention_masks[torch.tensor(test_indices)]

	train_level1_labels = level1_labels[torch.tensor(train_indices)]
	dev_level1_labels = level1_labels[torch.tensor(dev_indices)]
	test_level1_labels = level1_labels[torch.tensor(test_indices)]

	train_level2_labels = level1_labels[torch.tensor(train_indices)]
	dev_level2_labels = level1_labels[torch.tensor(dev_indices)]
	test_level2_labels = level1_labels[torch.tensor(test_indices)]

	num_pos_train_set = np.count_nonzero(train_level1_labels.cpu().numpy() == 1)
	train_set_size = train_input_ids.size()[0]
	total_dataset_size = train_input_ids.size()[0] + dev_input_ids.size()[0] + test_input_ids.size()[0]

	train_input_ids, train_attention_masks, train_level1_labels, train_level2_labels = \
	undersample(train_input_ids, train_attention_masks, train_level1_labels, train_level2_labels)
	# print(train_input_ids.size())
	# print(train_labels.size())
	train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_level1_labels, train_level2_labels)
	dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_level1_labels, dev_level2_labels)
	test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_level1_labels, test_level2_labels)

	batch_size = 32
	train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

	dev_dataloader = DataLoader(
            dev_dataset, # The dev samples.
            sampler = SequentialSampler(dev_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

	test_dataloader = DataLoader(
            test_dataset, # The test samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

	model = BertForSequenceClassification.from_pretrained(
    	"bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    	num_labels = 2, # The number of output labels--2 for binary classification.
    	output_attentions = False, # Whether the model returns attentions weights.
    	output_hidden_states = False, # Whether the model returns all hidden-states.
	)

	if torch.cuda.is_available():
		model = model.cuda()

	optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
          		)

	epochs = 10

	total_steps = len(train_dataloader) * epochs

	scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

	total_t0 = time.time()

	for epoch_i in range(0, epochs):
		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		print('Training...')

		t0 = time.time()

		total_train_loss = 0

		model.train()

		for step, batch in enumerate(train_dataloader):
			if step % 40 == 0 and not step == 0:
				elapsed = format_time(time.time() - t0)
				print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

			b_input_ids = batch[0]
			b_input_mask = batch[1]
			b_labels = batch[2]

			if torch.cuda.is_available():
				b_input_ids = b_input_ids.cuda()
				b_input_mask = b_input_mask.cuda()
				b_labels = b_labels.cuda()

			model.zero_grad()

			loss, logits = model(b_input_ids,
								token_type_ids=None,
								attention_mask=b_input_mask,
								labels=b_labels)

			total_train_loss += loss.item()

			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			optimizer.step()

			scheduler.step()

		avg_train_loss = total_train_loss / len(train_dataloader)

		training_time = format_time(time.time() - t0)

		print("")
		print("  Average training loss: {0:.2f}".format(avg_train_loss))
		print("  Training epcoh took: {:}".format(training_time))

	print("")
	print("Training complete!")

	print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

	print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))

	model.eval()

	predictions = []
	true_level1_labels = []
	true_level2_labels = []

	for batch in test_dataloader:
		b_input_ids = batch[0]
		b_input_mask = batch[1]
		b_level1_labels = batch[2]
		b_level2_labels = batch[3]

		if torch.cuda.is_available():
			b_input_ids = b_input_ids.cuda()
			b_input_mask = b_input_mask.cuda()
			b_level1_labels = b_level1_labels.cuda()
			b_level2_labels = b_level2_labels.cuda()


		with torch.no_grad():
			outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

		logits = outputs[0]

		logits = np.argmax(logits.detach().cpu().numpy(), 1).tolist()
		level1_label_ids = b_level1_labels.to('cpu').numpy().tolist()
		level2_label_ids = b_level2_labels.to('cpu').numpy().tolist()

		'''
		print(logits)
		print(level1_label_ids)
		print(level2_label_ids)
		'''

		predictions.extend(logits)
		true_level1_labels.extend(level1_label_ids)
		true_level2_labels.extend(level2_label_ids)

	'''
	print("predictions: ")
	print(predictions)
	print("true_level1_labels: ")
	print(true_level1_labels)
	print("true_level2_labels: ")
	print(true_level2_labels)
	'''

	_yield = compute_yield(true_level1_labels, predictions, num_pos_train_set)
	_burden = compute_burden(true_level2_labels, predictions, train_set_size, total_dataset_size)
	print("yield: %.4f" % _yield)
	print("burden: %.4f" % _burden)

def compute_yield(y_test_L1, y_pred, num_pos_train_set):
	true_pos = 0
	false_neg = 0
	for idx, label in enumerate(y_test_L1):
		if label == 1:
			if label == y_pred[idx]:
				true_pos += 1
			else:
				false_neg += 1

	return (true_pos + num_pos_train_set) / (true_pos + false_neg + num_pos_train_set)

def compute_burden(y_test_L2, y_pred, train_set_size, total_dataset_size):
	true_pos = 0
	false_pos = 0
	for idx, label in enumerate(y_pred):
		if label == 1:
			if label == y_test_L2[idx]:
				true_pos += 1
			else:
				false_pos += 1

	return (true_pos + false_pos + train_set_size) / (total_dataset_size)



def main():
	project = sys.argv[1]
	abstracts, level1_labels, level2_labels = get_project_abstracts_labels(project)
	input_ids, attention_masks, level1_labels, level2_labels = bert_tokenize(abstracts, level1_labels, level2_labels)
	train_bert(project, input_ids, attention_masks, level1_labels, level2_labels)

if __name__ == "__main__":
	main()