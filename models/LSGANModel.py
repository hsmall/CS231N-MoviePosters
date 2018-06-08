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

class LSGANModel(): 
        def __init__(self, genre, batch_size, z_dims=100, image_shape=(64, 64), alpha=0.2, reuse=True, lr=1e-3, beta1=0.5):
                tf.reset_default_graph()
                self.image_shape = image_shape
                self.genre = genre 
                self.learning_rate = lr
                self.batch_size = batch_size 
                self.z_dims = z_dims 

                print('batch size', batch_size)
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

        def fit(self, sess, name, num_epochs=5, checkpoint_directory=None, show_every=1, print_every=1, load=False):
                self.name = name
                self.total_g_sum = tf.summary.merge([self.z_summary, self.G_summary, self.G_loss_summary])
                self.total_d_sum = tf.summary.merge([self.z_summary, self.D_loss_summary])
                self.sess = sess 

                if checkpoint_directory is not None and load:
                    self.load_model(checkpoint_directory)
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
                filename = "output/" + self.genre + "/" + self.name + "/" + str(i) + '.jpg'
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

        def get_train_op(self, D_loss):
                D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
                return tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(D_loss, var_list=D_vars)

        def leaky_relu(self, x): 
                return tf.maximum(x, self.alpha*x)

        def get_loss(self, logits_real, logits_fake):
                D_loss_left = tf.subtract(logits_real, tf.random_uniform(tf.shape(logits_real), 0.8, 1.2))**2
                D_loss_right = logits_fake**2
                D_loss_left = 0.5 * tf.reduce_mean(D_loss_left) 
                D_loss_right = 0.5 * tf.reduce_mean(D_loss_right) 

                D_loss = D_loss_left + D_loss_right 

                return D_loss

    
        def __call__(self, x, training=True):
            with tf.variable_scope("discriminator"):
                logits = tf.layers.flatten(x)
                print(logits.get_shape())
                logits = tf.layers.dense(
                        inputs=logits, 
                        units=512)
                logits = tf.layers.batch_normalization(
                        inputs=logits, 
                        training=True) 
                logits = self.leaky_relu(logits) 

                print(logits.get_shape())

                logits = tf.layers.dense(
                        inputs=logits, 
                        units=256)
                logits = tf.layers.batch_normalization(
                        inputs=logits, 
                        training=True)
                logits = self.leaky_relu(logits) 

                print(logits.get_shape())

                logits = tf.layers.dense(
                        inputs=logits, 
                        units=64)
                logits = tf.layers.batch_normalization(
                        inputs=logits, 
                        training=True)
                logits = self.leaky_relu(logits) 

                print(logits.get_shape())

                logits = tf.layers.dense(
                        inputs=logits, 
                        units=1)

                print(logits.get_shape())
                return logits 

class Generator():
        def __init__(self, alpha=0.2, lr=1e-3, beta1=0.5, keep_prob=0.5): 
                self.alpha = alpha
                self.lr = lr
                self.beta1 = beta1
                self.keep_prob = keep_prob

        def leaky_relu(self, x): 
                #return tf.nn.leaky_relu(x, alpha=self.alpha) 
                return tf.maximum(x, self.alpha * x)

        def get_loss(self, logits_real, logits_fake):
                G_loss = tf.subtract(logits_fake, tf.ones_like(logits_fake))**2
                G_loss = 0.5 * tf.reduce_mean(G_loss)
                return G_loss

        def get_train_op(self, G_loss):
                G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
                return tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(G_loss, var_list=G_vars)

        def __call__(self, z, training=True):
            with tf.variable_scope("generator"):
                img = tf.layers.dense(
                        inputs=z, 
                        units=4096)
                img = tf.layers.batch_normalization(
                        inputs=img, 
                        training=True)
                img = tf.nn.relu(img)
                print("gen", img.get_shape())

                img = tf.layers.dense(
                        inputs=z, 
                        units=2048)
                img = tf.layers.batch_normalization(
                        inputs=img, 
                        training=True)
                img = tf.nn.relu(img) 
                print("gen", img.get_shape())

                img = tf.layers.dense(
                        inputs=z, 
                        units=512)
                img = tf.layers.batch_normalization(
                        inputs=img, 
                        training=True)
                img = tf.nn.relu(img) 
                print("gen", img.get_shape())

                img = tf.layers.dense(
                        inputs=z, 
                        units=64)
                img = tf.layers.batch_normalization(
                        inputs=img,
                        training=True)
                img = tf.nn.relu(img)
                print("gen", img.get_shape())

                img = tf.layers.dense(
                        inputs=z,
                        units=12288)
                img = tf.reshape(img,( -1, 64, 64, 3))
                img = tf.nn.tanh(img)
                print("gen", img.get_shape())
                return img
        
        def sampler(self, z):
            with tf.variable_scope('generator') as scope: 
                scope.reuse_variables()
                return self.common_alg(z, False)


