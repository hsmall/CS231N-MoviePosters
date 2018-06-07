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
        def __init__(self, genre, batch_size, z_dims=100, image_shape=(64, 64), alpha=0.2, reuse=True, lr=1e-3, beta1=0.5):
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

        def fit(self, sess, name, num_epochs=5, show_every=1, print_every=1, checkpoint_directory=None):
                self.name = name
                self.total_g_sum = tf.summary.merge([self.z_summary, self.G_summary, self.G_loss_summary])
                self.total_d_sum = tf.summary.merge([self.z_summary, self.D_loss_summary])
              
                if checkpoint_directory is not None: 
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
                # z = np.random.normal(0, 1, [self.batch_size, self.z_dims]).astype(np.float32)
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
                                                                
                                # _, G_loss_curr = sess.run(
                                #         [self.G_train_op, self.G_loss], 
                                #         {self.z: z})
                                _, G_loss_curr, g_summary = sess.run(
                                        [self.G_train_op, self.G_loss, self.total_g_sum], 
                                        {self.z: z})

                                # if (G_loss_curr > 2):
                                #     _, G_loss_curr = sess.run(
                                #             [self.G_train_op, self.G_loss], 
                                #             {self.z:z})


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
                # logits = self.cv1(x)
                # logits = self.relu1(logits)
                #
                # logits = self.cv2(logits)
                # logits = self.bn2(logits)
                # logits = self.relu2(logits) 
                #
                # logits = self.cv3(logits)
                # logits = self.bn3(logits)
                # logits = self.relu3(logits) 
                #
                # logits = self.cv4(logits)
                # logits = self.bn4(logits)
                # logits = self.relu4(logits)
                #
                # logits = self.cv5(logits)

                return logits 

        # def build_graph(self, images): 
        #         with tf.variable_scope('discriminator'): 
        #                 initializer = tf.truncated_normal_initializer(0.02)
        #
        #                 logits = tf.layers.conv2d(
        #                         inputs=images, 
        #                         filters=64,
        #                         kernel_size=5, 
        #                         strides=2,
        #                         padding='same',
        #                         activation=self.leaky_relu, 
        #                         name='conv2d_1',
        #                         kernel_initializer=initializer)
        #                 logits = self.leaky_relu(logits)
        #                 # logits = tf.layers.dropout(logits, rate=self.keep_prob)
        #                 print(logits.get_shape())
        #
        #
        #                 logits = tf.layers.conv2d(
        #                         inputs=logits, 
        #                         filters=128, 
        #                         kernel_size=5, 
        #                         strides=2, 
        #                         padding='same',
        #                         name='conv2d_2',
        #                         kernel_initializer=initializer)
        #                 logits = tf.layers.batch_normalization(
        #                         inputs=logits,
        #                         training=True)
        #                 logits = self.leaky_relu(logits)
        #                 # logits = tf.layers.dropout(logits, rate=self.keep_prob)
        #
        #                 logits = tf.layers.conv2d(
        #                         inputs=logits, 
        #                         filters=256, 
        #                         kernel_size=5, 
        #                         strides=2,
        #                         name='conv2d_3',
        #                         padding='same',
        #                         kernel_initializer=initializer) 
        #                 logits = tf.layers.batch_normalization(
        #                         inputs=logits, 
        #                         training=True)
        #                 logits = self.leaky_relu(logits)
        #                 # logits = tf.layers.dropout(logits, rate=self.keep_prob)
        #
        #                 logits = tf.layers.conv2d(
        #                         inputs=logits, 
        #                         filters=512, 
        #                         kernel_size=5, 
        #                         strides=2, 
        #                         padding='same',
        #                         name='conv2d_4',
        #                         kernel_initializer=initializer)
        #                 logits = tf.layers.batch_normalization(
        #                         inputs=logits, 
        #                         training=True)
        #                 logits = self.leaky_relu(logits)
        #                 # logits = tf.layers.dropout(logits, rate=self.keep_prob)
        #
        #                 # logits = tf.layers.conv2d(
        #                 #         inputs=logits, 
        #                 #         filters=2048, 
        #                 #         kernel_size=5, 
        #                 #         strides=2, 
        #                 #         padding='same', 
        #                 #         kernel_initializer=initializer)
        #                 # # logits = self.leaky_relu(logits)
        #                 # logits = tf.layers.batch_normalization(
        #                 #         inputs=logits, 
        #                 #         training=True)
        #                 # logits = self.leaky_relu(logits)
        #                 # logits = tf.layers.dropout(logits, rate=self.keep_prob)
        #                 # logits = self.leaky_relu(logits)
        #                 
        #                 logits = tf.layers.flatten(logits)
        #                 print(logits.get_shape())
        #                 
        #                 logits = tf.layers.dense(
        #                         inputs=logits, 
        #                         units=1)
        #                 return logits 
        #
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

                # img = tf.layers.conv2d_transpose(
                #         inputs=img, 
                #         filters=512, 
                #         kernel_size=4, 
                #         strides=2, 
                #         padding='same', 
                #         use_bias=False)
                # img = tf.layers.batch_normalization(
                #         inputs=img, 
                #         training=True)
                # img = tf.nn.relu(img) 
                #
                # print('conv1', img.get_shape())
                
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
                # img = self.dense1(z) 
                # img = self.bn1(img) 

                # img = self.relu1(img) 
                #
                # img = self.reshape1(img)
                #
                # img = self.cv1(img)
                # img = self.bn1(img)
                # img = self.relu1(img)
                #
                # img = self.cv2(img)
                # img = self.bn2(img)
                # img = self.relu2(img) 
                #
                # img = self.cv3(img)
                # img = self.bn3(img)
                # img = self.relu3(img) 
                #
                # img = self.cv4(img)
                # img = self.bn4(img)
                # img = self.relu4(img) 
                #
                # img = self.cv5(img)
                # img = self.tanh1(img)
                return img
        
        def sampler(self, z):
            with tf.variable_scope('generator') as scope: 
                scope.reuse_variables()
                return self.common_alg(z, False)

        # def common_alg(self, z, training):
        #     img = tf.layers.Conv2D(
        #             filters=512, 
        #             kernel_size=5, 
        #             strides=2, 
        #             padding='same')
        #     img = tf.layers.BatchNormalization()
        #     img = tf.layers.InputSpec

        # def common_alg(self, z, training):
        #                 initializer = tf.random_normal_initializer(0.02) 
        #
        #                 img = tf.layers.dense(
        #                         inputs=z, 
        #                         units=4*4*512) 
        #                 img = tf.reshape(img, (-1, 4, 4, 512))
        #
        #                 # img = tf.layers.batch_normalization(
        #                 #         inputs=img,
        #                 #         training=training)
        #                 # img = tf.nn.relu(img)
        #                 print("img dense", img.get_shape())
        #                 # img = tf.layers.conv2d_transpose(
        #                 #         inputs=img, 
        #                 #         filters=512, 
        #                 #         kernel_size=4, 
        #                 #         strides=2, 
        #                 #         padding='same', 
        #                 #         name='deconv_11', 
        #                 #         kernel_initializer=initializer)
        #                 # img = tf.layers.batch_normalization(
        #                 #         inputs =img, 
        #                 #         training=True)
        #                 # img = tf.nn.relu(img)
        #                 # print("first conv", img.get_shape())
        #                 img = tf.layers.conv2d_transpose(
        #                         inputs=img, 
        #                         filters=256, 
        #                         kernel_size=4, 
        #                         strides=2, 
        #                         padding='same',
        #                         name='deconv_1',
        #                         kernel_initializer=initializer)
        #                 # img = tf.nn.relu(img)
        #                 img = tf.layers.batch_normalization(
        #                         inputs=img, 
        #                         training=True)
        #                 img = tf.nn.relu(img)
        #                 # img = tf.layers.dropout(
        #                 #         img, 
        #                 #         rate=self.keep_prob) 
        #
        #                 img = tf.layers.conv2d_transpose(
        #                         inputs=img, 
        #                         filters=128, 
        #                         kernel_size=5, 
        #                         strides=2, 
        #                         padding='same', 
        #                         name='deconv_2',
        #                         kernel_initializer=initializer)
        #                 # img = tf.nn.relu(img)
        #                 img = tf.layers.batch_normalization(
        #                         inputs=img, 
        #                         training=training)
        #                 img = tf.nn.relu(img)
        #                 # img = tf.nn.relu(img)
        #                 # img = tf.layers.dropout(img, rate=self.keep_prob)
        #                 print("img conv1", img.get_shape())
        #                 # img = tf.layers.batch_normalization(
        #                 #       inputs=img, 
        #                 #       training=True) 
        #                 # print(img.get_shape())
        #
        #                 img = tf.layers.conv2d_transpose(
        #                         inputs=img, 
        #                         filters=64, 
        #                         kernel_size=5, 
        #                         strides=2, 
        #                         padding='same', 
        #                         name='deconv_3',
        #                         kernel_initializer=initializer)
        #                 # img = tf.nn.relu(img)
        #                 img = tf.layers.batch_normalization(
        #                         inputs=img, 
        #                         training=training)
        #                 img = tf.nn.relu(img) 
        #                 # img = tf.layers.dropout(img, rate=self.keep_prob)
        #                 print("img conv2", img.get_shape())
        #                 # img = tf.layers.batch_normalization(
        #                 #       inputs=img,
        #                 #       training=True)
        #                 # print(img.get_shape())
        #
        #                 # 4, 2 if 
        #                 # img = tf.layers.conv2d_transpose(
        #                 #         inputs=img, 
        #                 #         filters=128, 
        #                 #         kernel_size=5, 
        #                 #         strides=2, 
        #                 #         padding='same', 
        #                 #         kernel_initializer=initializer)
        #                 # # img = tf.nn.relu(img)
        #                 # img = tf.layers.batch_normalization(
        #                 #         inputs=img, 
        #                 #         training=training)
        #                 # img = tf.nn.relu(img)
        #                 #
        #                 # img = tf.layers.dropout(img, rate=self.keep_prob) 
        #                 # print("img conv3", img.get_shape())
        #                 # img = tf.layers.batch_normalization(
        #                 #       inputs=img, 
        #                 #       training=True) 
        #                 # print(img.get_shape())
        #
        #                 img = tf.layers.conv2d_transpose(
        #                         inputs=img, 
        #                         filters=3, 
        #                         kernel_size=5,
        #                         strides=2,
        #                         name='deconv_4',
        #                         padding='same',
        #                         kernel_initializer=initializer)
        #                 print("final image", img.get_shape().as_list())
        #                 # img = tf.layers.dropout(img, rate=self.keep_prob)
        #                 img = tf.nn.tanh(img)
        #                 # print(img.get_shape())
        #                 return img

