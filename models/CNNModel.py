from models.Model import Model

import numpy as np
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress Logging/Warnings

class CNNModel(Model):
    def __init__(self, genres, label_probs, image_shape, filter_counts,
        unit_counts, resize_shape=None):

        self.label_probs = label_probs
        self.image_shape = image_shape
        self.filter_counts = filter_counts
        self.unit_counts = unit_counts
        Model.__init__(self, genres)

    def build_graph(self, resize_shape):
        self.X_placeholder = tf.placeholder(tf.float32, [None] + list(self.image_shape))
        self.y_placeholder = tf.placeholder(tf.float32, [None, self.num_classes])
        self.learning_rate = tf.placeholder(tf.float32, shape=())
        self.reg = 0.0
        self.dropout = 0.25
        
        if resize_shape is None:
            conv_layers = [self.X_placeholder]
        else:
            conv_layers = [tf.image.resize_images(
                images = self.X_placeholder,
                size = resize_shape
            )]
        
        #filter_counts = [32, 32, 64, 64, 128, 128]
        filter_counts = [32, 32, 32, 32]
        for i in range(len(filter_counts)):
            conv_layers.append(self.build_conv_layer(
                inputs = conv_layers[-1],
                num_filters = filter_counts[i],
            ))
            
            conv_layers.append(self.build_conv_layer(
                inputs = conv_layers[-1],
                num_filters = filter_counts[i],
            ))

            if i % 2 == 0:
                #conv_layers.append(tf.layers.max_pooling2d(
                #   inputs = conv_layers[-1],
                #   pool_size = 2,
                #   strides = 2,
                #   padding = "valid"
                #))
                
                conv_layers.append(tf.contrib.layers.layer_norm(
                    inputs = conv_layers[-1],
                ))

        fc_layers = [tf.contrib.layers.flatten(conv_layers[-1])]
        unit_counts = [1024]
        #unit_counts = [512]
        for i in range(len(unit_counts)):
            fc_layers.append(self.build_fc_layer(
                inputs = fc_layers[-1],
                num_units = unit_counts[i]
            ))

        output = tf.layers.dense(
            inputs = fc_layers[-1],
            units = self.num_classes,
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            bias_initializer = tf.zeros_initializer(),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(self.reg),
            activation = tf.nn.sigmoid
        )
        
        return output

    def build_conv_layer(self, inputs, num_filters):
        return tf.layers.dropout(
            inputs = tf.layers.conv2d(
                inputs = inputs,
                filters = num_filters,
                kernel_size = 5,
                padding = "same",
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                bias_initializer = tf.zeros_initializer(),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(self.reg),
                activation = tf.nn.elu
            ),
            rate = self.dropout,
            training = self.is_training,
        )
    
    def build_fc_layer(self, inputs, num_units):
        return tf.layers.dropout(
            inputs = tf.layers.dense(
                inputs = inputs,
                units = num_units,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                bias_initializer = tf.zeros_initializer(),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(self.reg),
                activation = tf.nn.elu
            ),
            rate = self.dropout,
            training = self.is_training,
        )
    
    def loss_fn(self, y, output):
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

