import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from GenreDataset import GenreDataset 

class GANModel(): 
	def __init__(self, genre, batch_size, z_dims=19286, image_shape=(128, 128), alpha=0.2, reuse=True, lr=1e-3, beta1=0.5):
		## 185 x 278
		# 128 x 128
		# 32 x 69
		# tf.reset_default_graph()
		tf.reset_default_graph()
		self.image_shape = image_shape
		self.genre = genre 
		self.learning_rate = lr
		self.batch_size = batch_size 

		self.generator= Generator(genre, reuse, reuse, lr, beta1)
		self.discriminator = Discriminator(genre, alpha, reuse, lr, beta1)

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
		self.images = tf.placeholder(tf.float32, shape=(None, self.image_shape[0]*self.image_shape[1]*3))
	
	def get_loss(self, logits_real, logits_fake):
		D_loss = self.discriminator.get_loss(logits_real, logits_fake)
		G_loss = self.generator.get_loss(logits_real, logits_fake)
		return D_loss, G_loss

	def fit(self, sess, batches, num_epochs=5, show_every=1, print_every=1):
		for epoch in range(0, num_epochs):
			print('Epoch #{0} out of {1}: '.format(epoch, num_epochs))
			if epoch % show_every == 0:
				samples = sess.run(self.G_sample)
				fig = self.show_images(samples[:5])
				plt.show()
				print()
			for minibatch in batches: 
				minibatch = np.reshape(minibatch, (self.batch_size, -1))
				_, D_loss_curr = sess.run([self.D_train_op, self.D_loss], {self.images: minibatch})
				_, G_loss_curr = sess.run([self.G_train_step, self.G_loss])

			if epoch & print_every == 0: 
				print('Epoch: {}, D: {:.4}, G:{.4}'.format(epoch, D_loss_curr, G_loss_curr))
		print('Final images')
		samples = sess.run(self.G_sample)

		fig = self.show_images(samples[:5])
		plt.show()

	def show_images(self, X):
		images = np.array(X, np.uint32)
		images = np.reshape(X, (-1, 128, 128, 3))
		fig = plt.figure() 
		rows, columns = 1, 5
		fig = plt.figure()
		for i in range(1, rows*columns+1):
			fig.add_subplot(rows, columns, i)
			image = images[i-1]
			plt.imshow(image)
			plt.axis('off')

class Discriminator(): 
	def __init__(self, genre, alpha=0.2, reuse=tf.AUTO_REUSE, lr=1e-3, beta1=0.5): 
		self.genre = genre; 
		self.alpha = alpha
		self.reuse = reuse 
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
			initializer = tf.variance_scaling_initializer(scale=2.0)
			images = tf.reshape(images, (-1, 128, 128, 3))
			conv1_layer = tf.layers.conv2d(
				inputs=images, 
				filters=64, 
				kernel_size=5, 
				strides=1,
				padding="same", 
				kernel_initializer=initializer,
				activation=self.leaky_relu)

			conv2_layer = tf.layers.conv2d(
				inputs=conv1_layer,
				filters=48, 
				kernel_size=5,
				strides=1, 
				padding="same", 
				kernel_initializer=initializer,
				activation=self.leaky_relu) 
			# activation2 = tf.maximum()

			maxpool1_layer = tf.layers.max_pooling2d(
				inputs=conv2_layer, 
				pool_size=3, 
				strides=2, 
				padding="same")

			conv3_layer = tf.layers.conv2d(
				inputs=maxpool1_layer, 
				filters=32, 
				strides=1, 
				kernel_size=5, 
				padding="same", 
				kernel_initializer=initializer, 
				activation = self.leaky_relu)

			conv4_layer = tf.layers.conv2d(
				inputs=conv3_layer, 
				filters=48, 
				strides=1, 
				kernel_size=5,
				padding="same", 
				kernel_initializer=initializer, 
				activation = self.leaky_relu)

			maxpool2_layer = tf.layers.max_pooling2d(
				inputs=conv3_layer, 
				pool_size=3, 
				strides=2, 
				padding="same")

			flattened = tf.layers.flatten(maxpool2_layer)
			
			logits = tf.layers.dense(
				inputs=flattened, 
				units=4096)

			logits = tf.layers.dense(
				inputs=logits,
				units=3)

			return logits 

class Generator():
	def __init__(self, genre, alpha=0.2, reuse=tf.AUTO_REUSE, lr=1e-3, beta1=0.5): 
		self.genre = genre; 
		self.alpha = alpha
		self.reuse = reuse
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
			dense1_layer = tf.layers.dense(
				inputs=z,
				units=4096,
				activation=tf.nn.relu)

			batchnorm1_layer = tf.layers.batch_normalization(
				inputs=dense1_layer, training=True)

			dense2_layer = tf.layers.dense(
				inputs=batchnorm1_layer, 
				units=13248, 
				activation=tf.nn.relu)

			batchnorm2_layer = tf.layers.batch_normalization(
				inputs=dense2_layer, training=True)

			reshaped = tf.reshape(batchnorm2_layer, (-1, 8, 8, 207))
			
			conv1_layer = tf.layers.conv2d_transpose(
				inputs=reshaped, 
				filters=128, 
				kernel_size=4, 
				strides=4, 
				activation=tf.nn.relu, 
				padding='same')

			batchnorm3_layer = tf.layers.batch_normalization(
				inputs=conv1_layer, 
				training=True)

			conv2_layer = tf.layers.conv2d_transpose(
				inputs=batchnorm3_layer,
				filters=3, 
				kernel_size=4, 
				strides=4, 
				activation=tf.nn.tanh, 
				padding="same")

			img = tf.reshape(conv2_layer, (-1, 49152))

			return img

