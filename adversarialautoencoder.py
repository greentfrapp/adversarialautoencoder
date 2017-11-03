"""
Replicates the Adversarial Autoencoder architecture (Figure 1) from
Makhzani, Alireza, et al. "Adversarial autoencoders." arXiv preprint arXiv:1511.05644 (2015).
https://arxiv.org/abs/1511.05644
Refer to Appendix A.1 from paper for implementation details
"""

import argparse
import random
import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn import manifold
from sklearn.decomposition import PCA

from tensorflow.examples.tutorials.mnist import input_data

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# DenseNetwork class creates a network with defined nodes, activations and layer names
# Encoder, decoder, discriminator etc. networks are based on this
class DenseNetwork(object):
	def __init__(self, nodes_per_layer, activations_per_layer, names_per_layer, network_name):
		self.name = network_name
		self.layers = []
		for layer_no, layer_name in enumerate(names_per_layer):
			self.layers.append({
				"name": layer_name,
				"nodes": nodes_per_layer[layer_no],
				"activation": activations_per_layer[layer_no]
				})
		return None
	def forwardprop(self, input_tensor, reuse_variables=False):
		if reuse_variables:
			tf.get_variable_scope().reuse_variables()
		with tf.name_scope(self.name):
			tensor = input_tensor
			for layer in self.layers:
				tensor = tf.layers.dense(
					inputs=tensor,
					units=layer["nodes"],
					activation=layer["activation"],
					kernel_initializer=tf.truncated_normal_initializer(.0,.01),
					name=layer["name"])
			return tensor

# AdversarialAutoencoder class
# creates its own tf.Session() for training and testing
class AdversarialAutoencoder(object):
	def __init__(self, z_dim=2, batch_size=100, n_epochs=1000, results_folder='./Results'):

		# Create results_folder
		self.results_path = results_folder + '/AdversarialAutoencoder'
		if not os.path.exists(self.results_path):
			if not os.path.exists(results_folder):
				os.mkdir(results_folder)
			os.mkdir(self.results_path)

		# Download data
		self.mnist = input_data.read_data_sets('./Data', one_hot=True)

		# Parameters for everything
		self.img_width = 28
		self.img_height = 28
		self.img_dim = self.img_width * self.img_height
		self.z_dim = z_dim
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.real_prior_mean = 0.0
		self.real_prior_stdev = 5.0
		self.learning_rate = 0.001
		self.n_classes = 10

		# Initialize networks
		self.encoder = DenseNetwork(
			nodes_per_layer=[1000, 1000, self.z_dim],
			activations_per_layer=[tf.nn.relu, tf.nn.relu, None],
			names_per_layer=["encoder_dense_1", "encoder_dense_2", "encoder_output"],
			network_name="Encoder")
		self.decoder = DenseNetwork(
			nodes_per_layer=[1000, 1000, self.img_dim],
			activations_per_layer=[tf.nn.relu, tf.nn.relu, tf.nn.sigmoid],
			names_per_layer=["decoder_dense_1", "decoder_dense_2", "decoder_output"],
			network_name="Decoder")
		self.discriminator = DenseNetwork(
			nodes_per_layer=[1000, 1000, 1],
			activations_per_layer=[tf.nn.relu, tf.nn.relu, None],
			names_per_layer=["discriminator_dense_1", "discriminator_dense_2", "discriminator_output"],
			network_name="Discriminator")

		# Create tf.placeholder variables for inputs
		self.original_image = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.img_dim], name='original_image')
		self.target_image = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.img_dim], name='target_image')
		self.real_prior = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim], name='real_prior')
		self.sample_latent_vector = tf.placeholder(dtype=tf.float32, shape=[1, self.z_dim], name='sample_latent_vector')

		# Outputs from forwardproping networks 
		with tf.variable_scope(tf.get_variable_scope()):
			self.latent_vector = self.encoder.forwardprop(self.original_image)
			self.reconstruction = self.decoder.forwardprop(self.latent_vector)
			self.score_real_prior = self.discriminator.forwardprop(self.real_prior)
			self.score_fake_prior = self.discriminator.forwardprop(self.latent_vector, reuse_variables=True)
			self.sample_image = self.decoder.forwardprop(self.sample_latent_vector, reuse_variables=True)

		# Loss
		self.reconstruction_loss = tf.reduce_mean(tf.square(self.target_image - self.reconstruction))
		score_real_prior_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score_real_prior), logits=self.score_real_prior))
		score_fake_prior_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.score_fake_prior), logits=self.score_fake_prior))
		self.discriminator_loss = score_real_prior_loss + score_fake_prior_loss
		self.encoder_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score_fake_prior), logits=self.score_fake_prior))

		# Filtering the variables to be trained
		all_variables = tf.trainable_variables()
		self.discriminator_variables = [var for var in all_variables if 'discriminator' in var.name]
		self.encoder_variables = [var for var in all_variables if 'encoder' in var.name]

		# Training functions
		self.autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.reconstruction_loss)
		self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.discriminator_loss, var_list=self.discriminator_variables)
		self.encoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.encoder_loss, var_list=self.encoder_variables)
		
		# Things to save in Tensorboard
		self.input_images = tf.reshape(self.original_image, [-1, self.img_width, self.img_height, 1])
		self.generated_images = tf.reshape(self.reconstruction, [-1, self.img_width, self.img_height, 1])
		tf.summary.scalar(name="Autoencoder Loss", tensor=self.reconstruction_loss)
		tf.summary.scalar(name="Discriminator Loss", tensor=self.discriminator_loss)
		tf.summary.scalar(name="Encoder Loss", tensor=self.encoder_loss)
		tf.summary.histogram(name="Encoder Distribution", values=self.latent_vector)
		tf.summary.histogram(name="Real Distribution", values=self.real_prior)
		tf.summary.image(name="Input Images", tensor=self.input_images, max_outputs=10)
		tf.summary.image(name="Generated Images", tensor=self.generated_images, max_outputs=10)
		self.summary_op = tf.summary.merge_all()

		# Boilerplate Tensorflow stuff
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)
		self.saver = tf.train.Saver()

		return None

	# Creates the checkpoint folders
	def load_checkpoint_folders(self, z_dim, batch_size, n_epochs):
		folder_name = "/{0}_{1}_{2}_{3}_adversarial_autoencoder".format(
			datetime.datetime.now(),
			z_dim,
			batch_size,
			n_epochs)
		tensorboard_path = self.results_path + folder_name + '/tensorboard'
		saved_model_path = self.results_path + folder_name + '/saved_models/'
		log_path = self.results_path + folder_name + '/log'
		if not os.path.exists(self.results_path + folder_name):
			os.mkdir(self.results_path + folder_name)
			os.mkdir(tensorboard_path)
			os.mkdir(saved_model_path)
			os.mkdir(log_path)
		return tensorboard_path, saved_model_path, log_path

	# Samples a point from a normal distribution
	def generate_sample_prior(self, mean, stdev):
		return np.random.randn(self.batch_size, self.z_dim) * stdev + mean

	# Returns the losses for logging
	def get_loss(self, batch_x, z_real_dist):
		a_loss, d_loss, e_loss, summary = self.sess.run([self.reconstruction_loss, self.discriminator_loss, self.encoder_loss, self.summary_op], feed_dict={self.original_image:batch_x, self.target_image:batch_x, self.real_prior:z_real_dist})
		return (a_loss, d_loss, e_loss, summary)

	# Loads most recent saved model
	def load_last_saved_model(self, model_directory=None):
		if model_directory is None:
			all_results = os.listdir(self.results_path)
			all_results.sort()
			if tf.train.latest_checkpoint(self.results_path + '/' + all_results[-1] + '/saved_models/') is None:
				print("No saved model found.")
				quit()
			model_directory = self.results_path + '/' + all_results[-1] + '/saved_models/'
		self.saver.restore(self.sess, save_path=tf.train.latest_checkpoint(model_directory))
		return None

	# Train
	def train(self):
		self.step = 0
		self.tensorboard_path, self.saved_model_path, self.log_path = self.load_checkpoint_folders(self.z_dim, self.batch_size, self.n_epochs)
		self.writer = tf.summary.FileWriter(logdir=self.tensorboard_path, graph=self.sess.graph)

		for epoch in range(1, self.n_epochs + 1):
			n_batches = int(self.mnist.train.num_examples / self.batch_size)
			print("------------------Epoch {}/{}------------------".format(epoch, self.n_epochs))

			for batch in range(1, n_batches + 1):
				z_real_dist = self.generate_sample_prior(self.real_prior_mean, self.real_prior_stdev)
				batch_x, _ = self.mnist.train.next_batch(self.batch_size)

				autoencoder_learning_rate = 0.001
				discriminator_learning_rate = 0.001
				encoder_learning_rate = 0.001

				self.sess.run(self.autoencoder_optimizer, feed_dict={self.original_image:batch_x, self.target_image:batch_x})
				self.sess.run(self.discriminator_optimizer, feed_dict={self.original_image: batch_x, self.target_image: batch_x, self.real_prior: z_real_dist, })
				self.sess.run(self.encoder_optimizer, feed_dict={self.original_image: batch_x, self.target_image: batch_x})

				# Print log and write to log.txt every 50 batches
				if batch % 50 == 0:
					a_loss, d_loss, e_loss, summary = self.get_loss(batch_x, z_real_dist)
					self.writer.add_summary(summary, global_step=self.step)
					print("Epoch: {}, iteration: {}".format(epoch, batch))
					print("Autoencoder Loss: {}".format(a_loss))
					print("Discriminator Loss: {}".format(d_loss))
					print("Generator Loss: {}".format(e_loss))
					with open(self.log_path + '/log.txt', 'a') as log:
						log.write("Epoch: {}, iteration: {}\n".format(epoch, batch))
						log.write("Autoencoder Loss: {}\n".format(a_loss))
						log.write("Discriminator Loss: {}\n".format(d_loss))
						log.write("Generator Loss: {}\n".format(e_loss))

				self.step += 1

			self.saver.save(self.sess, save_path=self.saved_model_path, global_step=self.step)

		print("Model Trained!")
		print("Tensorboard Path: {}".format(self.tensorboard_path))
		print("Log Path: {}".format(self.log_path + '/log.txt'))
		print("Saved Model Path: {}".format(self.saved_model_path))
		return None

	# Generate a single sample image
	def generate_sample_image(self, sample_latent_vector=None, title=None):
		if sample_latent_vector is None:
			sample_latent_vector = np.zeros(self.z_dim)
		elif len(sample_latent_vector) < self.z_dim:
			print("Insufficient dimensions for latent vector, appending zeros...")
			sample_latent_vector = np.concatenate((sample_latent_vector, np.zeros(self.z_dim - len(sample_latent_vector))))
		elif len(sample_latent_vector) > self.z_dim:
			print("Too many dimensions for latent vector, shortening vector...")
			sample_latent_vector = sample_latent_vector[:self.z_dim]
		print("Generating image for latent vector: {}".format(sample_latent_vector))
		scale_x = 4.
		scale_y = 4.
		fig = plt.figure(figsize=(scale_x, scale_y))
		gs = gridspec.GridSpec(1, 1)
		z = np.reshape(sample_latent_vector, (1, self.z_dim))
		x = self.sess.run(self.sample_image, feed_dict={self.sample_latent_vector: z})
		ax = plt.Subplot(fig, gs[0])
		img = np.array(x.tolist()).reshape(self.img_width, self.img_height)
		ax.imshow(img, cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		if title is None:
			title = str(sample_latent_vector)
		ax.set_title(title)
		fig.add_subplot(ax)
		plt.show(block=False)
		return None
		
	# Generate a grid of sample images
	def generate_sample_image_grid(self, n_x=10, x_range=[-10, 10], n_y=10, y_range=[-10, 10]):
		if n_x == 1:
			step_x = 1
			x_range[1] = x_range[0] + 1
		else:
			step_x = (x_range[1] - x_range[0]) / (n_x - 1)
			# extend range so that np.arange is inclusive of x_range[1]
			x_range[1] += step_x
		if n_y == 1:
			step_y = 1
			y_range[1] = y_range[0] + 1
		else:
			step_y = (y_range[1] - y_range[0]) / (n_y - 1)
			# extend range so that np.arange is inclusive of y_range[1]
			y_range[1] += step_y

		x_points = np.arange(x_range[0], x_range[1], step_x).astype(np.float32)
		y_points = np.arange(y_range[0], y_range[1], step_y).astype(np.float32)[::-1] # reverses y_points so that graph is negative at bottom

		scale_x = 8. / n_x
		scale_y = 8. / n_y
		fig = plt.figure(figsize=(n_x*scale_x, n_y*scale_y))
		gs = gridspec.GridSpec(n_y, n_x, wspace=0.0, hspace=0.0)

		for i, g in enumerate(gs):
			z = np.concatenate(([x_points[int(i % n_x)]], [y_points[int(i / n_x)]], np.zeros(self.z_dim - 2)))
			z = np.reshape(z, (1, 2))
			x = self.sess.run(self.sample_image, feed_dict={self.sample_latent_vector: z})
			ax = plt.Subplot(fig, g)
			img = np.array(x.tolist()).reshape(self.img_width, self.img_height)
			ax.imshow(img, cmap='gray')
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_aspect('equal')
			fig.add_subplot(ax)
		plt.show(block=False)
		return None

	# Encodes images and plots the encodings
	def plot_latent_vectors(self, dataset_x=None, dataset_y=None, n_test=10000):
		if dataset_x is None or dataset_y is None:
			print('Loading {} images from MNIST test data'.format(n_test))
			dataset_x, dataset_y = self.mnist.test.next_batch(n_test)

		colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
		
		n_datapoints = len(dataset_x)
		fig, ax = plt.subplots()
		n_batch = int(n_datapoints / self.batch_size)
		batch_no = 0
		plot_data = {}

		while batch_no < n_batch:
			
			batch_x = dataset_x[batch_no * self.batch_size:(batch_no + 1) * self.batch_size]
			batch_y = dataset_y[batch_no * self.batch_size:(batch_no + 1) * self.batch_size]

			batch_z = self.sess.run(self.latent_vector, feed_dict={self.original_image: batch_x})
			
			for i, z in enumerate(batch_z):
				if batch_y[i].argmax() not in plot_data:
					plot_data[batch_y[i].argmax()] = {'x':[], 'y':[]}
				plot_data[batch_y[i].argmax()]['x'].append(z[0])
				plot_data[batch_y[i].argmax()]['y'].append(z[1])

			batch_no += 1

		for label, data in plot_data.items():
			ax.scatter(data['x'], data['y'], c=colors[label], label=label, edgecolors='none')

		ax.legend()
		plt.show(block=False)
		return None

	def plot_high_dim(self, dataset_x=None, dataset_y=None, n_test=10000, custom_latent_vectors=[]):
		if dataset_x is None or dataset_y is None:
			print('Loading {} images from MNIST test data'.format(n_test))
			dataset_x, dataset_y = self.mnist.test.next_batch(n_test, shuffle=False)

		colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#FFFFFF"]
		edgecolors = ["none", "none", "none", "none", "none", "none", "none", "none", "none", "none", "#000000"]

		n_datapoints = len(dataset_x)
		fig, ax = plt.subplots()
		n_batch = int(n_datapoints / self.batch_size)
		batch_no = 0
		all_z = None
		plot_data = {}

		while batch_no < n_batch:

			batch_x = dataset_x[batch_no * self.batch_size:(batch_no + 1) * self.batch_size]
			batch_y = dataset_y[batch_no * self.batch_size:(batch_no + 1) * self.batch_size]

			batch_z = self.sess.run(self.latent_vector, feed_dict={self.original_image: batch_x})

			if all_z is None:
				all_z = batch_z
			else:
				all_z = np.concatenate((all_z, batch_z))

			batch_no += 1

		pca = PCA(n_components=2, random_state=1)
		pca.fit(all_z)

		transformed_z = pca.transform(all_z)
		for i, z in enumerate(transformed_z):
			if dataset_y[i].argmax() not in plot_data:
				plot_data[dataset_y[i].argmax()] = {'x':[], 'y':[]}
			plot_data[dataset_y[i].argmax()]['x'].append(z[0])
			plot_data[dataset_y[i].argmax()]['y'].append(z[1])
		for label, data in plot_data.items():
			ax.scatter(data['x'], data['y'], c=colors[label], label=label, edgecolors=edgecolors[label])

		if len(custom_latent_vectors) > 0:
			transformed_vectors = pca.transform(custom_latent_vectors)
			custom_x = []
			custom_y = []
			for i, z in enumerate(transformed_vectors):
				custom_x.append(z[0])
				custom_y.append(z[1])
			ax.scatter(custom_x, custom_y, c=colors[-1], label='Custom', edgecolors=edgecolors[-1])

		ax.legend()
		plt.show(block=False)
		new_pca_vector = None
		while True:
			new_pca_vector = raw_input("New point: ")
			if new_pca_vector == "q":
				break
			self.pca2img(pca, eval(new_pca_vector))
		return None

	def pca2img(self, pca, pca_vector):
		pca_vector = np.array(pca_vector).reshape(1,2)
		z = pca.inverse_transform(pca_vector)
		self.generate_sample_image(sample_latent_vector=z[0], title=str(pca_vector))
		return None

def main():
	parser = argparse.ArgumentParser(
		description="Replicates the Adversarial Autoencoder architecture, refer to https://arxiv.org/abs/1511.05644",
		epilog="Use --train to train a model, followed by --sample/--samplegrid/--plot to generate sample images or plot encoded vectors.")
	parser.add_argument('--train', 
		action='store_true',
		default=False,
		help='Train a model')
	parser.add_argument('--sample', 
		action='store_true',
		default=False,
		help='Sample a single image')
	parser.add_argument('-z', '--latent_vector', 
		action='store',
		default=[0, 0],
		help='Sample latent vector (default [0,0])')
	parser.add_argument('--samplegrid', 
		action='store_true',
		default=False,
		help='Sample a grid of images')
	parser.add_argument('-rz1', '--range_z1', 
		action='store',
		default=[-10, 10],
		help='Range of z1 values (default [-10,10])')
	parser.add_argument('-rz2', '--range_z2', 
		action='store',
		default=[-10, 10],
		help='Range of z2 values (default [-10,10])')
	parser.add_argument('-nz1', '--no_steps_z1', 
		action='store',
		default=10,
		help='Number of z1 values (default 10)')
	parser.add_argument('-nz2', '--no_steps_z2', 
		action='store',
		default=10,
		help='Number of z2 values (default 10)')
	parser.add_argument('--plot', 
		action='store_true',
		default=False,
		help='Convert images to latent vectors and plot vectors')
	parser.add_argument('--plot_hi', 
		action='store_true',
		default=False,
		help='Convert images to latent vectors and plot vectors via t-SNE mapping')
	parser.add_argument('-i', '--no_images',
		action='store',
		default=10000,
		help='Number of images to plot (default 10000)')
	parser.add_argument('--custom_latent_vectors',
		action='store',
		default='[]',
		help='Set of latent vectors to map in t-SNE')
	parser.add_argument('--z_dim',
		action='store',
		default=2,
		help='Number of dimensions for latent vector')
	args = parser.parse_args()

	if args.train:
		model = AdversarialAutoencoder(z_dim=int(args.z_dim))
		model.train()
	elif args.sample:
		model = AdversarialAutoencoder(z_dim=int(args.z_dim))
		model.load_last_saved_model()
		model.generate_sample_image(sample_latent_vector=eval(args.latent_vector))
	elif args.samplegrid:
		if isinstance(args.range_z1, basestring):
			args.range_z1 = eval(args.range_z1)
		if isinstance(args.range_z2, basestring):
			args.range_z2 = eval(args.range_z2)
		model = AdversarialAutoencoder(z_dim=int(args.z_dim))
		model.load_last_saved_model()
		model.generate_sample_image_grid(
			n_x=int(args.no_steps_z1), 
			x_range=args.range_z1, 
			n_y=int(args.no_steps_z2), 
			y_range=args.range_z2)
	elif args.plot:
		model = AdversarialAutoencoder(z_dim=int(args.z_dim))
		model.load_last_saved_model()
		model.plot_latent_vectors(n_test=int(args.no_images))
	elif args.plot_hi:
		model = AdversarialAutoencoder(z_dim=int(args.z_dim))
		model.load_last_saved_model()
		model.plot_high_dim(n_test=int(args.no_images), custom_latent_vectors=eval(args.custom_latent_vectors))
	else:
		parser.print_help()
	raw_input("Hit Enter To Close")


if __name__ == "__main__":
	main()
	
