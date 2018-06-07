import numpy as np 
import datetime
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from models.util import Progbar
from GenreDataset import GenreDataset 
import os
import sys
from PIL import Image

class DCGANModel(): 
        def __init__(self, genre, batch_size,z_dims=100, image_shape=(64, 64), alpha=0.2, reuse=True, lr=1e-3, beta1=0.5):
                tf.reset_default_graph()
                self.image_shape = image_shape
                self.genre = genre 
                self.learning_rate = lr
                self.batch_size = batch_size 
                self.z_dims = z_dims 

                self.generator= Generator(alpha=alpha, lr=lr, beta1=beta1)
                self.discriminator = Discriminator(alpha=alpha, lr=lr, beta1=beta1)
                
                self.add_placeholders()
                
                self.G_sample = self.generator(self.z)
                self.z_summary = tf.summary.histogram("z", self.z)
                
                with tf.variable_scope("") as scope:
                        self.logits_real = self.discriminator(self.images)
                        scope.reuse_variables()
                        self.logits_fake  = self.discriminator(self.G_sample)

                self.D_loss, self.G_loss = self.get_loss(self.logits_real, self.logits_fake)
                self.D_loss_summary = tf.summary.scalar("Discriminator_Loss", self.D_loss)
                self.G_loss_summary = tf.summary.scalar("Generator_Loss", self.G_loss) 
                
                self.G_summary = tf.summary.image("G", self.G_sample)
                self.D_train_op = self.discriminator.get_train_op(self.D_loss)
                self.G_train_op = self.generator.get_train_op(self.G_loss)

                self.D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
                self.G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
                
                self.saver = tf.train.Saver()

        def add_placeholders(self): 
                self.z = tf.placeholder(tf.float32, shape=(None, self.z_dims))
                self.images = tf.placeholder(tf.float32, shape=(None, self.image_shape[0],self.image_shape[1], 3))
        
        def get_loss(self, logits_real, logits_fake):
                D_loss = self.discriminator.get_loss(logits_real, logits_fake)
                G_loss = self.generator.get_loss(logits_real, logits_fake)
                return D_loss, G_loss

        def fit(self, sess, name, num_epochs=5, show_every=1, print_every=1, checkpoint_directory=None, load=False):
                self.name = name
                self.total_g_sum = tf.summary.merge([self.z_summary, self.G_summary, self.G_loss_summary])
                self.total_d_sum = tf.summary.merge([self.z_summary, self.D_loss_summary])
                self.sess = sess

                if checkpoint_directory is not None and load:
                    checkpoint = tf.train.get_checkpoint_state(checkpoint_directory)
                    checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
                    self.saver.restore(self.sess, os.path.join(checkpoint_directory, checkpoint_name))
                    return
                elif checkpoint_directory is not None:
                    self.load_model(checkpoint_directory)
                    z = np.random.uniform(-1, 1, [self.batch_size, self.z_dims]).astype(np.float32)
                    samples = sess.run(self.G_sample, {self.z: z})
                    fig = self.show_images(samples[0:3])
                    plt.show()
                    print()
                    self.save_images(samples)
                    return 

                log_file = "./logs/" + name
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                self.writer = tf.summary.FileWriter(log_file, sess.graph)
                sess.run(tf.global_variables_initializer())
                
                counter = 1
                l_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dims]).astype(np.float32) 
                r_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dims]).astype(np.float32)
                
                z = self.linear_interpolation(l_z, r_z)
                for epoch in range(0, num_epochs):
                        batches = GenreDataset(self.genre, self.batch_size)
                        num_batches = batches.num_batches()+1
                        progbar = Progbar(target = num_batches)
                        print('Epoch #{0} out of {1}: '.format(epoch, num_epochs))
                        if epoch % show_every == 0:
                                samples = sess.run(
                                        self.G_sample,
                                        {self.z : z})
                                fig = self.show_images(samples[0:3], True)
                                plt.show()
                                print()
                                
                                ex = batches.get_batch(0)
                                idx = np.random.randint(0, ex.shape[0])
                                fig = self.show_images(ex[idx:idx+3], True)
                                plt.show()
                                print()
                        # for batch, minibatch in enumerate(batches): 
                        for batch_idx in range(0, num_batches-2):
                                minibatch = batches.get_batch(batch_idx)
                                
                                _, D_loss_curr, d_summary= sess.run(
                                        [self.D_train_op, self.D_loss, self.total_d_sum], 
                                        {self.images: minibatch, self.z: z})
                                                                
                                _, G_loss_curr, g_summary = sess.run(
                                        [self.G_train_op, self.G_loss, self.total_g_sum], 
                                        {self.z: z})


                                self.writer.add_summary(d_summary, counter) 
                                self.writer.add_summary(g_summary, counter)
                                counter += 1
                                progbar.update(batch_idx+1, [('D Loss', D_loss_curr), ('G Loss', G_loss_curr)])

                        if epoch % print_every == 0: 
                             print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))
                        
                        if epoch % 10 == 0:
                            save_file = 'checkpoint/'+self.genre+'/'+name+'/'
                            os.makedirs(os.path.dirname(save_file), exist_ok=True)
                            self.saver.save(sess, save_file, global_step=epoch)
                
                print('Final images')
                samples = sess.run(self.G_sample, {self.z: z})
            
                fig = self.show_images(samples[:5])
                plt.show()
                print() 
                self.save_images(samples)
    
        def linear_interpolation(self, left, right): 
            line = np.linspace(0, 1, self.batch_size) 
            noise = np.zeros((self.batch_size, self.z_dims))
            print(line.shape, left.shape, right.shape)
            for i in range(0, self.batch_size):
                noise[i] = (left[i] * line[i] + right[i] * (1-line[i]))
            return noise

        def load_model(self, directory):
            checkpoint = tf.train.get_checkpoint_state(directory)
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(directory, checkpoint_name))

        def save_images(self, X, edit=True):
            if (edit):
                X += 1
                X /= 2.0 
            for i in range(1, X.shape[0]+1):
                image = X[i-1] 
                image *= 255 
                image = image.astype('uint8')
                im = Image.fromarray(image) 
                filename = "output/" + self.genre + "/" + self.name + "/" + str(i) + ".jpg"
                os.makedirs(os.path.dirname(filename), exist_ok=True) 
                im.save(filename)

        def show_images(self, X, edit=True):
                fig = plt.figure()
                if (edit): 
                    X += 1
                    X /= 2.0
                rows, columns = 1, X.shape[0]
                fig = plt.figure()
                for i in range(1, rows*columns+1):
                        fig.add_subplot(rows, columns, i)
                        image = X[i-1]
                        plt.imshow(image)
                        plt.axis('off')

class Discriminator(): 
        def __init__(self, alpha=0.2, lr=1e-3, beta1=0.5, keep_prob=0.5): 
                self.alpha = alpha
                self.lr = lr
                self.beta1 = beta1
                self.keep_prob = keep_prob

                self.build_graph()

        def get_train_op(self, D_loss):
                D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
                return tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(D_loss, var_list=D_vars)

        def leaky_relu(self, x): 
                return tf.maximum(x, self.alpha*x)

        def get_loss(self, logits_real, logits_fake):
                print(logits_real)
                D_loss_left = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.random_uniform(tf.shape(logits_real), 0.8, 1.2), logits=logits_real)
                D_loss_right = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.random_uniform(tf.shape(logits_fake), 0, 0.2), logits=logits_fake)
                D_loss_left = tf.reduce_mean(D_loss_left)
                D_loss_right = tf.reduce_mean(D_loss_right)
                D_loss = D_loss_left + D_loss_right
                return D_loss

        def build_graph(self):
            with tf.variable_scope(''):
                initializer = tf.truncated_normal_initializer(0.02)
                self.cv1 = tf.layers.Conv2D(
                        filters=64, 
                        kernel_size=5, 
                        strides=2, 
                        kernel_initializer=initializer,
                        padding="same")
                self.relu1 = self.leaky_relu 

                self.cv2 = tf.layers.Conv2D(
                        filters=128, 
                        kernel_size=5, 
                        strides=2, 
                        kernel_initializer=initializer,
                        padding='same')
                self.bn2 = tf.layers.batch_normalization 
                self.relu2 = self.leaky_relu 

                self.cv3 = tf.layers.Conv2D(
                        filters=256, 
                        kernel_size=5, 
                        strides=2, 
                        kernel_initializer=initializer,
                        padding='same')
                self.bn3 = tf.layers.batch_normalization 
                self.relu3 = self.leaky_relu 

                self.cv4 = tf.layers.Conv2D(
                        filters=512, 
                        kernel_size=5, 
                        strides=2, 
                        kernel_initializer=initializer,
                        padding='same')
                self.bn4 = tf.layers.batch_normalization
                self.relu4 = self.leaky_relu 

                self.cv5 = tf.layers.Conv2D(
                        filters=1, 
                        kernel_size=4, 
                        strides=2, 
                        kernel_initializer=initializer,
                        padding='same')
    
        def __call__(self, x, training=True):
            with tf.variable_scope("discriminator"):
                logits = tf.layers.conv2d(
                        inputs=x, 
                        filters=64, 
                        kernel_size=4, 
                        strides=2, 
                        padding="same", 
                        use_bias=False)
                logits = self.leaky_relu(logits) 

                logits = tf.layers.conv2d(
                        inputs=logits, 
                        filters=128, 
                        kernel_size=4, 
                        strides=2, 
                        padding="same", 
                        use_bias=False) 
                logits = tf.layers.batch_normalization(
                        inputs=logits, 
                        training=True)
                logits = self.leaky_relu(logits) 

                logits = tf.layers.conv2d(
                        inputs=logits, 
                        filters=256, 
                        kernel_size=4, 
                        strides=2, 
                        padding="same", 
                        use_bias=False)
                logits = tf.layers.batch_normalization(
                        inputs=logits, 
                        training=True)
                logits = self.leaky_relu(logits) 

                logits = tf.layers.conv2d(
                        inputs=logits, 
                        filters=512, 
                        kernel_size=4, 
                        strides=2, 
                        padding="same", 
                        use_bias=False)
                logits = tf.layers.batch_normalization(
                        inputs=logits, 
                        training=True)
                logits = self.leaky_relu(logits) 

                logits = tf.layers.conv2d(
                        inputs=logits, 
                        filters=1, 
                        kernel_size=4, 
                        strides=2, 
                        padding="same", 
                        use_bias=False)

                print("logits", logits.get_shape())

                return logits 

class Generator():
        def __init__(self, alpha=0.2, lr=1e-3, beta1=0.5, keep_prob=0.5): 
                self.alpha = alpha
                self.lr = lr
                self.beta1 = beta1
                self.keep_prob = keep_prob

                self.build_graph()

        def leaky_relu(self, x): 
                #return tf.nn.leaky_relu(x, alpha=self.alpha) 
                return tf.maximum(x, self.alpha * x)

        def get_loss(self, logits_real, logits_fake):
                G_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake)
                G_loss = tf.reduce_mean(G_loss)
                return G_loss

        def get_train_op(self, G_loss):
                G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
                return tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(G_loss, var_list=G_vars)

        # 16384
        def build_graph(self):
                with tf.variable_scope(''):
                    initializer = tf.random_normal_initializer(0.02)
                    
                    self.dense1 = tf.layers.Dense(
                            units=4*4*512)
                    self.reshape1 = tf.keras.layers.Reshape((4, 4, 512))

                    self.cv1 = tf.layers.Conv2DTranspose(
                            filters=512, 
                            kernel_size=5, 
                            strides=2, 
                            padding="same",
                            kernel_initializer=initializer)
                    self.bn1 = tf.layers.batch_normalization
                    self.relu1 = tf.nn.relu

                    self.cv2 = tf.layers.Conv2DTranspose(
                            filters=256, 
                            kernel_size=5, 
                            strides=2, 
                            padding="same",
                            kernel_initializer=initializer)
                    self.bn2 = tf.layers.batch_normalization
                    self.relu2 = tf.nn.relu 

                    self.cv3 = tf.layers.Conv2DTranspose(
                            filters=128, 
                            kernel_size=5, 
                            strides=2, 
                            padding="same",
                            kernel_initializer=initializer)
                    self.bn3 = tf.layers.batch_normalization
                    self.relu3 = tf.nn.relu

                    self.cv4 = tf.layers.Conv2DTranspose(
                            filters=64, 
                            kernel_size=5, 
                            strides=2, 
                            padding='same',
                            kernel_initializer=initializer)
                    self.bn4 = tf.layers.batch_normalization
                    self.relu4 = tf.nn.relu 
                    
                    self.cv5 = tf.layers.Conv2DTranspose(
                            filters=3, 
                            kernel_size=5, 
                            strides=2, 
                            padding="same",
                            kernel_initializer=initializer)
                    self.tanh1 = tf.nn.tanh 
        def __call__(self, z, training=True):
            with tf.variable_scope("generator"):
                img = tf.layers.dense(
                        inputs=z, 
                        units=4*4*512)
                img = tf.layers.batch_normalization(
                        inputs=img, 
                        training=True)
                img = tf.nn.relu(img) 

                img = tf.reshape(img, (-1, 4, 4, 512))

                print('img', img.get_shape())

                img = tf.layers.conv2d_transpose(
                        inputs=img, 
                        filters=256, 
                        kernel_size=4, 
                        strides=2, 
                        padding='same', 
                        use_bias=False)
                img = tf.layers.batch_normalization(
                        inputs=img, 
                        training=True)
                img = tf.nn.relu(img) 

                print('conv2', img.get_shape())

                img = tf.layers.conv2d_transpose(
                        inputs=img, 
                        filters=128, 
                        kernel_size=4, 
                        strides=2, 
                        padding='same', 
                        use_bias=False)
                img = tf.layers.batch_normalization(
                        inputs=img, 
                        training=True)
                img = tf.nn.relu(img) 

                print('conv3', img.get_shape())

                img = tf.layers.conv2d_transpose(
                        inputs=img, 
                        filters=64, 
                        kernel_size=4, 
                        strides=2, 
                        padding='same', 
                        use_bias=False)
                img = tf.layers.batch_normalization(
                        inputs=img, 
                        training=True)
                img = tf.nn.relu(img)

                print('conv4', img.get_shape())

                img = tf.layers.conv2d_transpose(
                        inputs=img, 
                        filters=3, 
                        kernel_size=4, 
                        strides=2, 
                        padding='same', 
                        use_bias=False)

                print('conv5 output', img.get_shape())
                img = tf.nn.tanh(img)
                return img
        
        def sampler(self, z):
            with tf.variable_scope('generator') as scope: 
                scope.reuse_variables()
                return self.common_alg(z, False)

