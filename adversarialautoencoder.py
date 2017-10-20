"""
Replicates the Adversarial Autoencoder architecture (Figure 1) from
Makhzani, Alireza, et al. "Adversarial autoencoders." arXiv preprint arXiv:1511.05644 (2015).
https://arxiv.org/abs/1511.05644
Refer to Appendix A.1 from paper for implementation details
"""

import argparse
import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec

from tensorflow.examples.tutorials.mnist import input_data


# DenseNetwork class creates a network with defined nodes, activations and layer names
# Encoder, decoder, discriminator etc. networks are based on this
class DenseNetwork:
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
class AdversarialAutoencoder:
	def __init__(self, batch_size=100, n_epochs=1000, results_folder='./Results'):

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
		self.z_dim = 2
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.learning_rate = 0.001
		self.beta1 = 0.9
		self.real_prior_mean = 0.0
		self.real_prior_stdev = 5.0

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
		self.autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.reconstruction_loss)
		self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.discriminator_loss, var_list=self.discriminator_variables)
		self.encoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.encoder_loss, var_list=self.encoder_variables)
		
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
	def load_checkpoint_folders(self, z_dim, learning_rate, batch_size, n_epochs, beta1):
		folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_adversarial_autoencoder".format(
			datetime.datetime.now(),
			z_dim,
			learning_rate,
			batch_size,
			n_epochs,
			beta1)
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
	def generate_sample_prior(self, mean, var):
		return np.random.randn(self.batch_size, self.z_dim) * var + mean

	# Returns the losses for logging
	def get_loss(self):
		z_real_dist = self.generate_sample_prior(self.real_prior_mean, self.real_prior_stdev)
		batch_x, _ = self.mnist.train.next_batch(self.batch_size)
		a_loss, d_loss, e_loss, summary = self.sess.run([self.reconstruction_loss, self.discriminator_loss, self.encoder_loss, self.summary_op], feed_dict={self.original_image:batch_x, self.target_image:batch_x, self.real_prior:z_real_dist})
		return (a_loss, d_loss, e_loss, summary)

	# Loads most recent saved model
	def load_last_saved_model(self, model_directory=None):
		if model_directory is None:
			all_results = os.listdir(self.results_path)
			all_results.sort()
			if tf.train.latest_checkpoint(self.results_path + '/' + all_results[-1] + '/Saved_models/') is None:
				print("No saved model found.")
				quit()
			model_directory = self.results_path + '/' + all_results[-1] + '/Saved_models/'
		self.saver.restore(self.sess, save_path=tf.train.latest_checkpoint(model_directory))
		return None

	# Train
	def train(self):
		self.step = 0
		self.tensorboard_path, self.saved_model_path, self.log_path = self.load_checkpoint_folders(self.z_dim, self.learning_rate, self.batch_size, self.n_epochs, self.beta1)
		self.writer = tf.summary.FileWriter(logdir=self.tensorboard_path, graph=self.sess.graph)

		for epoch in range(1, self.n_epochs + 1):
			n_batches = int(self.mnist.train.num_examples / self.batch_size)
			print("------------------Epoch {}/{}------------------".format(epoch, self.n_epochs))

			for batch in range(1, n_batches + 1):
				z_real_dist = self.generate_sample_prior(self.real_prior_mean, self.real_prior_stdev)
				batch_x, _ = self.mnist.train.next_batch(self.batch_size)
				self.sess.run(self.autoencoder_optimizer, feed_dict={self.original_image:batch_x, self.target_image:batch_x})
				self.sess.run(self.discriminator_optimizer, feed_dict={self.original_image: batch_x, self.target_image: batch_x, self.real_prior: z_real_dist})
				self.sess.run(self.encoder_optimizer, feed_dict={self.original_image: batch_x, self.target_image: batch_x})

				# Print log and write to log.txt every 50 batches
				if batch % 50 == 0:
					a_loss, d_loss, e_loss, summary = self.get_loss()
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
	def generate_sample_image(self, sample_latent_vector=[0., 0.]):
		self.generate_sample_image_grid(
			n_x=1, 
			x_range=[sample_latent_vector[0], sample_latent_vector[0] + 1.],
			n_y=1,
			y_range=[sample_latent_vector[1], sample_latent_vector[1] + 1.])
		
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

		for i,g in enumerate(gs):
			z = np.concatenate(([x_points[int(i % n_x)]], [y_points[int(i / n_x)]]))
			z = np.reshape(z, (1, 2))
			x = self.sess.run(self.sample_image, feed_dict={self.sample_latent_vector: z})
			ax = plt.Subplot(fig, g)
			img = np.array(x.tolist()).reshape(self.img_width, self.img_height)
			ax.imshow(img, cmap='gray')
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_aspect('equal')
			fig.add_subplot(ax)
		plt.show()

	# Encodes images and plots the encodings
	def plot_latent_vectors(self, dataset_x=None, dataset_y=None, n_test=10000):
		if dataset_x is None or dataset_y is None:
			print('Loading {} images from MNIST test data'.format(n_test))
			dataset_x, dataset_y = self.mnist.test.next_batch(n_test)

		colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
		
		n_datapoints = len(dataset_x)
		plotted_datapoints = 0
		fig, ax = plt.subplots()
		last_batch = False
		while not last_batch:
			if n_datapoints - plotted_datapoints < 2 * self.batch_size:
				last_batch = True
			if last_batch:
				batch_x = dataset_x[plotted_datapoints:]
				batch_y = dataset_y[plotted_datapoints:]
			else:
				batch_x = dataset_x[plotted_datapoints:plotted_datapoints + self.batch_size]
				batch_y = dataset_y[plotted_datapoints:plotted_datapoints + self.batch_size]

			z = self.sess.run(self.latent_vector, feed_dict={self.original_image: batch_x})
			labeled_data = {}
			for data_no, datapoint in enumerate(z):
				if batch_y[data_no].argmax() not in labeled_data:
					labeled_data[batch_y[data_no].argmax()] = {'x':[], 'y':[]}
				labeled_data[batch_y[data_no].argmax()]['x'].append(datapoint[0])
				labeled_data[batch_y[data_no].argmax()]['y'].append(datapoint[1])
			for label in labeled_data:
				if last_batch:
					ax.scatter(labeled_data[label]['x'], labeled_data[label]['y'], c=colors[label], label=label, edgecolors='none')
				else:
					ax.scatter(labeled_data[label]['x'], labeled_data[label]['y'], c=colors[label], edgecolors='none')
			plotted_datapoints += self.batch_size
		ax.legend()
		plt.show()

if __name__ == "__main__":
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
	parser.add_argument('-i', '--no_images',
		action='store',
		default=10000,
		help='Number of images to plot (default 10000)')
	args = parser.parse_args()

	if args.train:
		model = AdversarialAutoencoder()
		model.train()
	elif args.sample:
		model = AdversarialAutoencoder()
		model.load_last_saved_model()
		model.generate_sample_image(sample_latent_vector=args.latent_vector)
	elif args.samplegrid:
		if isinstance(args.range_z1, basestring):
			args.range_z1 = eval(args.range_z1)
		if isinstance(args.range_z2, basestring):
			args.range_z2 = eval(args.range_z2)
		model = AdversarialAutoencoder()
		model.load_last_saved_model()
		model.generate_sample_image_grid(
			n_x=int(args.no_steps_z1), 
			x_range=args.range_z1, 
			n_y=int(args.no_steps_z2), 
			y_range=args.range_z2)
	elif args.plot:
		model = AdversarialAutoencoder()
		model.load_last_saved_model()
		model.plot_latent_vectors(n_test=int(args.no_images))
	else:
		parser.print_help()
