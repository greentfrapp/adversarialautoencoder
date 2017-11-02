"""
Demonstrates the following baseline anomaly-detection algorithms via sklearn
- One-class SVM
- Robust covariance estimate
- Isolation Forest
- Local Outlier Factor
The dataset is similar to the one described in Section 5.3 from 
Zhou, Chong, and Randy C. Paffenroth. "Anomaly Detection with Robust Deep Autoencoders." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2017.
https://doi.org/10.1145/3097983.3098052
By default, 
5000 datapoints are sampled from the MNIST dataset
95% (4750) consist of images of the digit '4'
5% (250) consist of anomalous images of other digits {'0', '7', '9'}
Reproducible sampling is ensured via setting a random seed and shuffle=False
"""

import numpy as np
import random
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from tensorflow.examples.tutorials.mnist import input_data

random_seed = np.random.RandomState(1)

classifiers = {
	"One-class SVM": svm.OneClassSVM(
		random_state=random_seed),
	"Robust covariance": EllipticEnvelope(
		contamination=0.05,
		random_state=random_seed),
	"Isolation Forest": IsolationForest(
		contamination=0.05,
		random_state=random_seed),
	"Local Outlier Factor": LocalOutlierFactor(
		contamination=0.05)
}

# shuffle with fixed seed
# need to call random.seed(seed) everytime to reinitialize
def shuffle(data, seed=1):
	random.seed(seed)
	random.shuffle(data)
	return data

# sample nominal and anomalous data from the MNIST dataset
def generate_mnist_anomaly_data(contamination=0.05, n_data=5000):
	mnist = input_data.read_data_sets('./Data', one_hot=True)
	nominal_label = {4}
	anomalous_label = {0, 7, 9}
	nominal_training_data = {'data':[], 'labels':[]}
	anomalous_training_data = {'data':[], 'labels':[]}

	total_training_size = n_data
	anomalous_training_size = int(contamination * total_training_size)
	nominal_training_size = total_training_size - anomalous_training_size

	print("Generating {} total datapoint(s)...".format(total_training_size))
	print("Generating {} nominal datapoint(s)...".format(nominal_training_size))
	print("Generating {} anomalous datapoint(s)...".format(anomalous_training_size))

	while len(nominal_training_data['data']) < nominal_training_size or len(anomalous_training_data['data']) < anomalous_training_size:
		
		sample_data, sample_label = mnist.train.next_batch(1, shuffle=False)
		sample_label = [[np.argmax(sample_label[0])]]

		if len(nominal_training_data['data']) < nominal_training_size and sample_label[0][0] in nominal_label:
			if len(nominal_training_data['data']) == 0:
				nominal_training_data['data'] = sample_data
				nominal_training_data['labels'] = sample_label
			else:
				nominal_training_data['data'] = np.concatenate((nominal_training_data['data'], sample_data))
				nominal_training_data['labels'] = np.concatenate((nominal_training_data['labels'], sample_label))

		elif len(anomalous_training_data['data']) < anomalous_training_size and sample_label[0][0] in anomalous_label:
			if len(anomalous_training_data['data']) == 0:
				anomalous_training_data['data'] = sample_data
				anomalous_training_data['labels'] = sample_label
			else:
				anomalous_training_data['data'] = np.concatenate((anomalous_training_data['data'], sample_data))
				anomalous_training_data['labels'] = np.concatenate((anomalous_training_data['labels'], sample_label))

	training_data = np.concatenate((nominal_training_data['data'], anomalous_training_data['data']))
	#training_labels = np.concatenate((nominal_training_data['labels'], anomalous_training_data['labels']))
	# label = 1 for nominal and label = -1 for anomalous
	training_labels = np.concatenate((np.ones(nominal_training_size), np.ones(anomalous_training_size) * -1))

	return shuffle(training_data), shuffle(training_labels)


def main():
	
	training_data, training_labels = generate_mnist_anomaly_data()
	# call fit and predict and print percentage error
	for classifier_name, classifier in classifiers.items():
		if classifier_name == "Local Outlier Factor":
			predicted_labels = classifier.fit_predict(training_data)
		else:
			classifier.fit(training_data)
			predicted_labels = classifier.predict(training_data)
		n_errors = (predicted_labels != training_labels).sum()
		percent_error = 100. * n_errors / total_training_size
		print("{} percentage error: {}".format(classifier_name, percent_error))

if __name__ == "__main__":
	main()