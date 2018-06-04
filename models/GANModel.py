import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from models.util import Progbar
from GenreDataset import GenreDataset 

class GANModel(): 
	def __init__(self, genre, batch_size, z_dims=100, image_shape=(64, 64), alpha=0.2, reuse=True, lr=1e-3, beta1=0.5):
		## 185 x 278
		# 128 x 128
		# 32 x 69
		# tf.reset_default_graph()
		tf.reset_default_graph()
		self.image_shape = image_shape
		self.genre = genre 
		self.learning_rate = lr
		self.batch_size = batch_size 

		self.generator= Generator(alpha=alpha, lr=lr, beta1=beta1)
		self.discriminator = Discriminator(alpha=alpha, lr=lr, beta1=beta1)

		z = tf.random_uniform([batch_size, z_dims], -1, 1)
		self.G_sample = self.generator.build_graph(z)
		
		with tf.variable_scope("") as scope:
			self.add_placeholders()
			self.logits_real = self.discriminator.build_graph(self.images)
			scope.reuse_variables()
			self.logits_fake = self.discriminator.build_graph(self.G_sample)

		self.D_loss, self.G_loss = self.get_loss(self.logits_real, self.logits_fake)

		self.D_train_op = self.discriminator.get_train_op(self.D_loss)
		self.G_train_op = self.generator.get_train_op(self.G_loss)

		self.D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
		self.G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

	def add_placeholders(self): 
		self.images = tf.placeholder(tf.float32, shape=(None, self.image_shape[0], self.image_shape[1], 3))
	
	def get_loss(self, logits_real, logits_fake):
		D_loss = self.discriminator.get_loss(logits_real, logits_fake)
		G_loss = self.generator.get_loss(logits_real, logits_fake)
		return D_loss, G_loss

	def fit(self, sess, batches, num_epochs=5, show_every=1, print_every=1):
		num_batches = batches.num_batches()+1
		for epoch in range(0, num_epochs):
			progbar = Progbar(target = num_batches)
			print('Epoch #{0} out of {1}: '.format(epoch, num_epochs))
			if epoch % show_every == 0:
				# samples = sess.fun(self.G_sample)
				samples = sess.run(self.G_sample)
				# print(samples[:1])
				fig = self.show_images(samples[:3])
				plt.show()
				print()
			for batch, minibatch in enumerate(batches): 
				# minibatch = np.reshape(minibatch, (self.batch_size, -1))
				_, D_loss_curr = sess.run([self.D_train_op, self.D_loss], {self.images: minibatch})
				_, G_loss_curr = sess.run([self.G_train_op, self.G_loss])
				progbar.update(batch+1, [('D Loss', D_loss_curr), ('G Loss', G_loss_curr)])

			if epoch & print_every == 0: 
				print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))
		
		print('Final images')
		samples = sess.run(self.G_sample)

		fig = self.show_images(samples[:3])
		plt.show()

	def show_images(self, X):
		images = np.array(X, np.uint32)
		images = np.reshape(X, (-1, 64, 64, 3))
		fig = plt.figure() 
		rows, columns = 1, X.shape[-1]
		fig = plt.figure()
		for i in range(1, rows*columns+1):
			fig.add_subplot(rows, columns, i)
			image = images[i-1]
			plt.imshow(image)
			plt.axis('off')

class Discriminator(): 
	def __init__(self, alpha=0.2, lr=1e-3, beta1=0.5): 
		self.alpha = alpha
		self.lr = lr
		self.beta1 = beta1

	def get_train_op(self, D_loss):
		D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
		return tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(D_loss, var_list=D_vars)

	def leaky_relu(self, x): 
		return tf.nn.leaky_relu(x, alpha=self.alpha) 

	def get_loss(self, logits_real, logits_fake):
		D_loss_left = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real)
		D_loss_right = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake)
		D_loss_left = tf.reduce_mean(D_loss_left)
		D_loss_right = tf.reduce_mean(D_loss_right)
		D_loss = D_loss_left + D_loss_right
		return D_loss

	def build_graph(self, images): 
		with tf.variable_scope('discriminator'): 
			initializer = tf.truncated_normal_initializer(0.02)
			print(images.get_shape())
			logits = tf.layers.conv2d(
				inputs=images, 
				filters=64, 
				kernel_size=5, 
				strides=2,
				padding='same', 
				activation=self.leaky_relu,
				kernel_initializer=initializer)
			print(logits.get_shape())

			logits = tf.layers.conv2d(
				inputs=logits, 
				filters=128, 
				kernel_size=5, 
				strides=2, 
				padding='same', 
				activation=self.leaky_relu,
				kernel_initializer=initializer) 
			logits = tf.layers.batch_normalization(
				inputs=logits, 
				training=True) 

			print(logits.get_shape())

			logits = tf.layers.conv2d(
				inputs=logits, 
				filters=256, 
				kernel_size=5, 
				strides=2, 
				padding='same',
				activation=self.leaky_relu,
				kernel_initializer=initializer) 
			logits = tf.layers.batch_normalization(
				inputs=logits, 
				training=True) 
			print(logits.get_shape())

			logits = tf.layers.conv2d(
				inputs=logits, 
				filters=512, 
				kernel_size=5, 
				strides=2, 
				padding='same', 
				activation=self.leaky_relu,
				kernel_initializer=initializer) 
			logits = tf.layers.batch_normalization(
				inputs=logits, 
				training=True)

			print(logits.get_shape())

			logits = tf.layers.flatten(logits)
			print(logits.get_shape())

			w = tf.get_variable('w', shape=[logits.get_shape().as_list()[-1], 1], initializer=tf.random_normal_initializer(0.02))
			# logits = tf.layers.dense(
			# 	inputs=logits, 
			# 	units=1, 
			# 	activation=tf.nn.sigmoid)
			b = tf.get_variable('b', shape=[1], initializer=tf.zeros_initializer())
			logits = tf.nn.xw_plus_b(logits, w, b)
			print(logits.get_shape())
			logits = tf.sigmoid(logits)
			return logits 

class Generator():
	def __init__(self, alpha=0.2, lr=1e-3, beta1=0.5): 
		self.alpha = alpha
		self.lr = lr
		self.beta1 = beta1

	def leaky_relu(self, x): 
		return tf.nn.leaky_relu(x, alpha=self.alpha) 

	def get_loss(self, logits_real, logits_fake):
		G_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake)
		G_loss = tf.reduce_mean(G_loss)
		return G_loss

	def get_train_op(self, G_loss):
		G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
		return tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(G_loss, var_list=G_vars)

	# 16384
	def build_graph(self, z):
		with tf.variable_scope('generator'):
			initializer = tf.random_normal_initializer(0.02)
			w = tf.get_variable('w', shape=[z.get_shape().as_list()[0], 4*4*512], initializer=tf.random_normal_initializer(0.02))
			# logits = tf.layers.dense(
			# 	inputs=logits, 
			# 	units=1, 
			# 	activation=tf.nn.sigmoid)
			b = tf.get_variable('b', shape=[4*4*512], initializer=tf.zeros_initializer())
			img = tf.nn.xw_plus_b(z, w, b)
			# img = tf.layers.batch_normalization(
			# 	inputs=img, 
			# 	training=True) 
			print(img.get_shape())

			img = tf.reshape(img, (-1, 4, 4, 512))
			print(img.get_shape())

			img = tf.layers.conv2d_transpose(
				inputs=img, 
				filters=512, 
				kernel_size=5, 
				strides=1, 
				padding='same', 
				activation=tf.nn.relu,
				kernel_initializer=initializer)
			# img = tf.layers.batch_normalization(
			# 	inputs=img,
			# 	training=True) 
			print(img.get_shape())

			img = tf.layers.conv2d_transpose(
				inputs=img, 
				filters=256, 
				kernel_size=5, 
				strides=2, 
				padding='same', 
				activation=tf.nn.relu,
				kernel_initializer=initializer) 
			# img = tf.layers.batch_normalization(
			# 	inputs=img, 
			# 	training=True) 
			print(img.get_shape())

			img = tf.layers.conv2d_transpose(
				inputs=img, 
				filters=128, 
				kernel_size=5, 
				strides=2, 
				padding='same', 
				activation=tf.nn.relu,
				kernel_initializer=initializer)
			# img = tf.layers.batch_normalization(
			# 	inputs=img,
			# 	training=True)
			print(img.get_shape())

			img = tf.layers.conv2d_transpose(
				inputs=img, 
				filters=64, 
				kernel_size=5, 
				strides=2, 
				padding='same', 
				activation=tf.nn.relu,
				kernel_initializer=initializer) 
			# img = tf.layers.batch_normalization(
			# 	inputs=img, 
			# 	training=True) 
			print(img.get_shape())

			img = tf.layers.conv2d_transpose(
				inputs=img, 
				filters=3, 
				kernel_size=5, 
				strides=2, 
				padding='same',
				activation=tf.nn.tanh,
				kernel_initializer=initializer) 
			print(img.get_shape())
			return img

