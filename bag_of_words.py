import sys
import pickle
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.base import clone
from joblib import Parallel, delayed
from scipy.stats import mode

class ClassImbalanceRedux:
	def __init__(self, clf, n_bags = 30):
		self.clf = clf
		self.n_bags = n_bags

	def fit(self, X, y, n_jobs = -1, seed = 93):
		# Fix seed for reproducibility of results
		np.random.seed(seed)
		# Check which class is imbalance
		counts = [np.sum(y == -1), np.sum(y == 1)]
		# self.under = np.argmin(counts)
		# self.under_count = counts[self.under]
		self.over = -1
		self.under_count = counts[1]
		self.fit_clfs = Parallel(n_jobs = n_jobs, verbose = 11)(delayed(self._fitBag)(X, y) for i in range(self.n_bags))

	def _fitBag(self, X, y):
		# Clean clf parameters
		clf = clone(self.clf)
		# Generate bootstrapped sample
		y_under_indexes = (y != self.over)
		y_under = y[y_under_indexes]
		y_over_indexes = np.random.choice(np.where(y == self.over)[0], self.under_count)
		y_over = y[y_over_indexes]
		X_bag = np.concatenate((X[y_under_indexes], X[y_over_indexes]), axis = 0)
		y_bag = np.concatenate((y_under, y_over), axis = 0)
		# Fit model in bag
		clf.fit(X_bag, y_bag)
		return clf

	def _predBag(self, clf, X_test):
		# Predict clf class
		self.pred_bag += 1
		print("Predicting for model in bag {}/{}".format(self.pred_bag, self.n_bags))
		return clf.predict(X_test)

	def predict(self, X_test, threshold = 0.5):
		self.pred_bag = 0
		self.y_hat = np.squeeze([self._predBag(clf, X_test) for clf in self.fit_clfs])
		#prob = np.sum(self.y_hat, axis = 0)/self.y_hat.shape[0]
		#return prob > threshold
		result, _ = mode(self.y_hat, axis=0)
		result = result.ravel()
		return result

	def predict_proba(self, X_test):
		self.pred_bag = 0
		self.y_hat = np.squeeze([self._predBag(clf, X_test) for clf in self.fit_clfs])
		prob = np.sum(self.y_hat, axis = 0)/self.y_hat.shape[0]
		return prob
	
	def save(self, path = '.'):
		np.save(path, arr = self)


def bag_of_words(project):
	project_file = "full_exports_level1_level2_labels/" + str(project) + "/labels_" + str(project) + ".csv"
	train_indices = pickle.load(open("full_exports_level1_level2_labels/" + str(project) + "/train_indices.pkl", "rb"))
	# print(project_file)
	dev_indices = pickle.load(open("full_exports_level1_level2_labels/" + str(project) + "/dev_indices.pkl", "rb"))
	test_indices = pickle.load(open("full_exports_level1_level2_labels/" + str(project) + "/test_indices.pkl", "rb"))
	
	train_indices_size = len(train_indices)
	train_indices = train_indices[0:train_indices_size]

	dataset = pd.read_csv(project_file, encoding="ISO-8859-1", converters={'level2_labels':str})
	train_data = dataset.loc[train_indices, :]
	dev_data = dataset.loc[dev_indices, :]
	test_data = dataset.loc[test_indices, :]

	train_processed_abstracts = []
	train_level1_labels = []
	train_level2_labels = []
	for abstract in tqdm(train_data['abstract']):
		processed_abstract = preprocess_abs(abstract)
		# print(processed_abstract)
		train_processed_abstracts.append(processed_abstract)	
	for label in tqdm(train_data['level1_labels']):
		if label == 0:
			train_level1_labels.append(1)
		else:
			train_level1_labels.append(label)
	for label in tqdm(train_data['level2_labels']):
		if label == "NA":
			train_level2_labels.append(-1)
		else:
			train_level2_labels.append(1)

	dev_processed_abstracts = []
	dev_level1_labels = []
	dev_level2_labels = []
	for abstract in tqdm(dev_data['abstract']):
		processed_abstract = preprocess_abs(abstract)
		# print(processed_abstract)
		dev_processed_abstracts.append(processed_abstract)
	for label in tqdm(dev_data['level1_labels']):
		if label == 0:
			dev_level1_labels.append(1)
		else:
			dev_level1_labels.append(label)
	for label in tqdm(dev_data['level2_labels']):
		if label == "NA":
			dev_level2_labels.append(-1)
		else:
			dev_level2_labels.append(1)

	test_processed_abstracts = []
	test_level1_labels = []
	test_level2_labels = []
	for abstract in tqdm(test_data['abstract']):
		processed_abstract = preprocess_abs(abstract)
		# print(processed_abstract)
		test_processed_abstracts.append(processed_abstract)
	for label in tqdm(test_data['level1_labels']):
		if label == 0:
			test_level1_labels.append(1)
		else:
			test_level1_labels.append(label)
	for label in tqdm(test_data['level2_labels']):
		if label == "NA":
			test_level2_labels.append(-1)
		else:
			test_level2_labels.append(1)

	#print(dev_level1_labels)
	#print(dev_level2_labels)



	matrix = CountVectorizer(max_features=20000)
	X_all = matrix.fit_transform(train_processed_abstracts + test_processed_abstracts).toarray()
	X_train = X_all[:len(train_processed_abstracts)]
	X_test = X_all[len(train_processed_abstracts):]
	y_train_L1 = train_level1_labels
	y_test_L1 = test_level1_labels
	y_train_L2 = train_level2_labels
	y_test_L2 = test_level2_labels
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	y_train_L1 = np.array(y_train_L1)
	y_test_L1 = np.array(y_test_L1)
	y_train_L2 = np.array(y_train_L2)
	y_test_L2 = np.array(y_test_L2)
	num_pos_train_set = np.count_nonzero(y_train_L1 == 1)
	train_set_size = len(X_train)
	total_dataset_size = len(train_processed_abstracts) + len(dev_processed_abstracts) + len(test_processed_abstracts)

	print("classifying...")

	print("Logistic Regression:")
	classifier = LogisticRegression(random_state = 0)
	cir = ClassImbalanceRedux(classifier)
	cir.fit(X_train, y_train_L1)
	y_pred = cir.predict(X_test)
	# print(classification_report(y_test, y_pred))
	_yield = compute_yield(y_test_L1, y_pred, num_pos_train_set)
	_burden = compute_burden(y_test_L2, y_pred, train_set_size, total_dataset_size)
	print("yield: %.4f" % _yield)
	print("burden: %.4f" % _burden)
	print("SVM:")
	classifier = SVC(kernel="linear", C=0.025, max_iter=1000)
	cir = ClassImbalanceRedux(classifier)
	cir.fit(X_train, y_train_L1)
	y_pred = cir.predict(X_test)	
	# print(classification_report(y_test, y_pred))
	_yield = compute_yield(y_test_L1, y_pred, num_pos_train_set)
	_burden = compute_burden(y_test_L2, y_pred, train_set_size, total_dataset_size)
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


def preprocess_abs(abstract):
	# print(abstract)
	abstract = re.sub('[^A-Za-z]', ' ', str(abstract))
	abstract = abstract.lower()

	tokenized_abstract = word_tokenize(abstract)
	for word in tokenized_abstract:
		if word in stopwords.words("english"):
			tokenized_abstract.remove(word)
	
	stemmer = PorterStemmer()
	for i in range(len(tokenized_abstract)):
		tokenized_abstract[i] = stemmer.stem(tokenized_abstract[i])
	

	processed_abstract = " ".join(tokenized_abstract)
	# print(processed_abstract)

	return processed_abstract

def main():
	project = sys.argv[1]
	bag_of_words(project)

if __name__ == "__main__":
	main()