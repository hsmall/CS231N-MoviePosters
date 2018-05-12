from models.Model import Model

import tensorflow as tf
import numpy as np 

class LogisticRegression(Model):
	def __init__(self, genres, label_probs, image_shape):
		self.label_probs = label_probs
		self.image_shape = image_shape
		Model.__init__(self, genres)

	def build_graph(self):
		self.X_placeholder = tf.placeholder(tf.float32, shape=[None] + list(self.image_shape))
		self.y_placeholder = tf.placeholder(tf.float32, shape=[None, self.num_classes])
		self.learning_rate = tf.placeholder(tf.float32, shape=[])

		input_dim = self.image_shape[0]*self.image_shape[1]*self.image_shape[2]
		self.W = tf.Variable(tf.random_normal(shape=[input_dim, self.num_classes]))
		self.b = tf.Variable(tf.zeros(shape=[1, self.num_classes]))

		X_flattened = tf.contrib.layers.flatten(self.X_placeholder)
		return tf.nn.sigmoid(tf.matmul(X_flattened, self.W) + self.b)

	# Multiclass Cross-Entropy Loss
	def loss_fn(self, y, output):
		#return -tf.reduce_mean(y*tf.log(tf.clip_by_value(output, 1e-10, 1.0)) + (1-y)*tf.log(tf.clip_by_value(1-output, 1e-10, 1.0)))
        
		pos_cross_entropy = y / self.label_probs * tf.log(tf.clip_by_value(output, 1e-10, 1.0))
		neg_cross_entropy = (1-y) / (1-self.label_probs) * tf.log(tf.clip_by_value(1-output, 1e-10, 1.0))
		return -tf.reduce_mean(pos_cross_entropy + neg_cross_entropy)

	def optimizer(self, loss):
		return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

	def predict(self, X, batch_size=None):
		if batch_size is None: batch_size = len(X)

		preds = np.zeros((len(X), self.num_classes))
		for i in range(0, len(X), batch_size): 
			preds[i:i+batch_size] = np.round(self.session.run(self.output, {
				self.X_placeholder : X[i:i+batch_size],
				self.is_training : False
			}))
		return preds

	def get_weights(self):
		return self.session.run(self.W)