import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import csv
from sklearn.metrics import precision_recall_fscore_support
vector_file = 'glove.6B.300d.txt'#'small_vector.txt'

train_file = 'datasets/dataset_lex/train.tsv'#'small_train.tsv'
test_file = 'datasets/dataset_lex/test.tsv'#'small_test.tsv'


# extract vectors into dictionary
word_dict = {}
with open(vector_file) as f:
	for line in f:
		entry = line.split()
		key = entry.pop(0).lower()
		word_dict[key] = list(map(float, entry))

# create train and test vectors


def file_to_data(file_name):
	data = []
	labels = []
	with open(file_name) as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		for line in tsvreader:
			ent1, ent2, rel = line
			if ent1 in word_dict and ent2 in word_dict:
				x = word_dict[ent1] + word_dict[ent2]
				y = int(rel == 'True')
				data.append(x)
				labels.append(y)
	return (np.array(data), np.array(labels))

train_x, train_y = file_to_data(train_file)
test_x, test_y = file_to_data(test_file)


classifier = learn.DNNClassifier(hidden_units=[10, 7], n_classes=2)

classifier.fit(train_x, train_y.T, steps=75, batch_size=100)
pred = classifier.predict(test_x)

xor = np.logical_xor(pred, test_y)
total = len(pred)
correct = total - np.sum(xor)
accuracy = correct / total
print(correct, total, accuracy)

print(precision_recall_fscore_support(test_y, pred))