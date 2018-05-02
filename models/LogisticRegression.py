from models.Model import Model

import progressbar  # pip install progressbar2
import tensorflow as tf
import numpy as np 

class LogisticRegression(Model):
	def __init__(self, genres, input_dim, output_dim):
		self.input_dim = input_dim
		self.output_dim = output_dim
		Model.__init__(self, genres)

	def build_graph(self):
		self.X_placeholder = tf.placeholder(tf.float32, shape=[None, self.input_dim])
		self.y_placeholder = tf.placeholder(tf.float32, shape=[None, self.output_dim])
		self.learning_rate = tf.placeholder(tf.float32, shape=[])

		self.W = tf.Variable(tf.random_normal(shape=[self.input_dim, self.output_dim]))
		self.b = tf.Variable(tf.zeros(shape=[1, self.output_dim]))

		return tf.nn.sigmoid(tf.matmul(self.X_placeholder, self.W) + self.b)

	# Multiclass Cross-Entropy Loss
	def loss_fn(self, y, output):
		return -tf.reduce_mean(y*tf.log(tf.clip_by_value(output, 1e-10, 1.0)) + (1-y)*tf.log(tf.clip_by_value(1-output, 1e-10, 1.0)))

	def optimizer(self, loss):
		return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

	def predict(self, X):
		return np.round(self.session.run(self.output, {
			self.X_placeholder : X
		}))

	def get_weights(self):
		return self.session.run(self.W)