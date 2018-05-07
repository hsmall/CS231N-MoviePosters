from models.Model import Model

import numpy as np
import tensorflow as tf
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress Logging/Warnings

class CNNModel(Model):
	def __init__(self, genres, image_shape):
		self.image_shape = image_shape
		Model.__init__(self, genres)

	def build_graph(self):
		self.X_placeholder = tf.placeholder(tf.float32, [None] + list(self.image_shape))
		self.y_placeholder = tf.placeholder(tf.float32, [None, self.num_classes])
		self.learning_rate = tf.placeholder(tf.float32, shape=())

		conv_layers = [self.X_placeholder]
		conv_layers.append(tf.layers.average_pooling2d(
			inputs = conv_layers[-1],
			pool_size = 4,
			strides = 4,
			padding = "valid"
		))

		for i in range(2):
			conv_layers.append(tf.layers.conv2d(
				inputs = conv_layers[-1],
				filters = 32,
				kernel_size = 3,
				padding = "same",
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.elu
			))

		#(278, 185, 3),
		conv_layers.append(tf.layers.max_pooling2d(
			inputs = conv_layers[-1],
			pool_size = 4,
			strides = 4,
			padding = "valid"
		))

		fc_layers = [tf.contrib.layers.flatten(conv_layers[-1])]
		for i in range(1):
			fc_layers.append(tf.layers.dense(
				inputs = fc_layers[-1],
				units = 1024,
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.zeros_initializer(),
				activation = tf.nn.elu
			))

		output = tf.layers.dense(
			inputs = fc_layers[-1],
			units = self.num_classes,
			kernel_initializer = tf.contrib.layers.xavier_initializer(),
			bias_initializer = tf.zeros_initializer(),
			activation = None
		)
		self.sigmoid_output = tf.nn.sigmoid(output)
		
		return output

	# Multiclass Cross-Entropy Loss
	def loss_fn(self, y, output):
		label_probs = tf.reduce_mean(y, axis=0, keepdims=True)
		pos_cross_entropy = y / label_probs * tf.log(tf.clip_by_value(output, 1e-10, 1.0))
		neg_cross_entropy = (1-y) / (1-label_probs) * tf.log(tf.clip_by_value(1-output, 1e-10, 1.0))
		return -tf.reduce_mean(pos_cross_entropy + neg_cross_entropy)

	def optimizer(self, loss):
		return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

	def predict(self, X):
		return np.round(self.session.run(self.sigmoid_output, {
			self.X_placeholder : X
		}))

