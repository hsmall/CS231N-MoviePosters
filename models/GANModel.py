import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from models.util import Progbar
from GenreDataset import GenreDataset 
import os 

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
                self.z_dims = z_dims 

                self.generator= Generator(alpha=alpha, lr=lr, beta1=beta1)
                self.discriminator = Discriminator(alpha=alpha, lr=lr, beta1=beta1)
                
                self.add_placeholders()
                z = tf.random_uniform([batch_size, z_dims], -1, 1)
                # z = tf.truncated_normal([batch_size, z_dims], 0, 0.5)
                self.G_sample = self.generator.build_graph(z)
                self.z_summary = tf.summary.histogram("z", z)
                
                with tf.variable_scope("") as scope:
                       # self.G_sample = self.generator.build_graph(self.z)
                        self.logits_real, self.logits_real_raw = self.discriminator.build_graph(self.images)
                        #self.sampler = self.generator.sampler(self.z)
                        scope.reuse_variables()
                        self.logits_fake, self.logits_fake_raw = self.discriminator.build_graph(self.G_sample)

                self.D_loss, self.G_loss = self.get_loss(self.logits_real_raw, self.logits_real,  self.logits_fake_raw, self.logits_fake)
                # print(self.logits_real, self.logits_real_raw,C self.D_loss)
                # print(self.logits_fake, self.logits_fake_raw, self.G_loss)
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
        
        def get_loss(self, logits_real_raw, logits_real, logits_fake_raw, logits_fake):
                D_loss = self.discriminator.get_loss(logits_real_raw, logits_real,  logits_fake_raw, logits_fake)
                G_loss = self.generator.get_loss(logits_real_raw, logits_real, logits_fake_raw, logits_fake)
                return D_loss, G_loss
        def inverse_transform(self, images): 
            return (images+1.)/2

        def fit(self, sess, num_epochs=5, show_every=1, print_every=1):
                self.total_g_sum = tf.summary.merge([self.z_summary, self.G_summary, self.G_loss_summary])
                self.total_d_sum = tf.summary.merge([self.z_summary, self.D_loss_summary])
                #merged = tf.summary.merge_all()
                self.writer = tf.summary.FileWriter("./logs", sess.graph)
                sess.run(tf.global_variables_initializer())
#                sess.run(tf.local_variables_initializer())
                counter = 1
                z = np.random.uniform(-1, 1, [self.batch_size, self.z_dims]).astype(np.float32) 

                for epoch in range(0, num_epochs):
                        batches = GenreDataset(self.genre, self.batch_size)
                        num_batches = batches.num_batches()+1
                        progbar = Progbar(target = num_batches)
                        print('Epoch #{0} out of {1}: '.format(epoch, num_epochs))
                        if epoch % show_every == 0:
                                samples = sess.run(
                                        self.G_sample,
                                        {self.z : z})
                                fig = self.show_images(samples[20:23], True)
                                plt.show()
                                print()
                                fig = self.show_images(samples[30:33], True) 
                                plt.show()
                                print()

                                fig=self.show_images(samples[10:13], True)
                                plt.show()
                                print() 

                                ex = batches.get_batch(0) 
                                fig = self.show_images(ex[20:23], True)
                                plt.show()
                                print()


                                # fig = self.show_images(batches.X[:3])
                                # plt.show()
                                # print()
                        # for batch, minibatch in enumerate(batches): 
                        for batch_idx in range(0, num_batches-2):
                                minibatch = batches.get_batch(batch_idx)
                                # print(minibatch.shape)
                                # minibatch = np.reshape(minibatch, (self.batch_size, -1))
                                _, D_loss_curr, d_summary= sess.run(
                                        [self.D_train_op, self.D_loss, self.total_d_sum], 
                                        {self.images: minibatch, self.z: z})
                                # _, D_loss_curr, d_summary = sess.run(
                                #         [self.D_train_op, self.D_loss, self.total_d_sum], 
                                #         {self.images: minibatch, self.z: z})
                                _, G_loss_curr, g_summary  = sess.run(
                                        [self.G_train_op, self.G_loss, self.total_g_sum],
                                        {self.z: z})
                                _, G_loss_curr, g_summary = sess.run(
                                        [self.G_train_op, self.G_loss, self.total_g_sum], 
                                        {self.z: z})

                                self.writer.add_summary(d_summary, counter) 
                                self.writer.add_summary(g_summary, counter)
                                counter += 1
                                progbar.update(batch_idx+1, [('D Loss', D_loss_curr), ('G Loss', G_loss_curr)])

                        if epoch & print_every == 0: 
                             print('Epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, D_loss_curr, G_loss_curr))
                        
                        if epoch & print_every == 0: 
                            self.saver.save(sess, os.path.join('checkpoint/'+self.genre, 'DCGAN'), global_step=epoch)
                
                print('Final images')
                z = np.random.uniform(0, 1, [5, self.z_dims])
                samples = sess.run(self.G_sample, {self.z: z})
            
                # self.writter.add_summary(samples, counter) 

                fig = self.show_images(samples[:5])
                plt.show()
                print() 
                self.save_images(samples)

        def save_images(self, X, edit=True):
            if (edit):
                X += 1
                X /= 2.0 
            for i in range(1, X.shape[0]+1):
                image = X[i-1] 
                im = Image.fromaray(image) 
                filename = "output/" + self.genre + i
                os.makedirs(os.path.dirname(filename), exist_ok=True) 
                im.save(filename)


        def show_images(self, X, edit=True):
                fig = plt.figure()
                if (edit): 
                    # X += 1
                    X += 1
                    X /= 2.0
                    # X += 1
                # print(X[0][0])
                rows, columns = 1, X.shape[0]
                fig = plt.figure()
                # print(X)
                for i in range(1, rows*columns+1):
                        fig.add_subplot(rows, columns, i)
                        image = X[i-1]
                        # min_val = np.min(image)
                        # max_val = np.max(image) 
                        # image = (image-min_val) / (max_val-min_val)
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
                #return tf.nn.leaky_relu(x, alpha=self.alpha) 
                return tf.maximum(x, self.alpha*x)

        def get_loss(self, logits_real_raw, logits_real, logits_fake_raw, logits_fake):
                D_loss_left = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real_raw), logits=logits_real_raw)
                D_loss_right = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake_raw), logits=logits_fake_raw)
                D_loss_left = tf.reduce_mean(D_loss_left)
                D_loss_right = tf.reduce_mean(D_loss_right)
                D_loss = D_loss_left + D_loss_right
                return D_loss

        def build_graph(self, images): 
                with tf.variable_scope('discriminator'): 
                        initializer = tf.truncated_normal_initializer(0.02)
                        # print(images.get_shape())
                        logits = tf.layers.conv2d(
                                inputs=images, 
                                filters=64, 
                                kernel_size=5, 
                                strides=2,
                                padding='same', 
                                kernel_initializer=initializer)
                        logits = self.leaky_relu(logits)
                        print(logits.get_shape())


                        logits = tf.layers.conv2d(
                                inputs=logits, 
                                filters=128, 
                                kernel_size=5, 
                                strides=2, 
                                padding='same', 
                                kernel_initializer=initializer) 
                        logits = tf.layers.batch_normalization(
                                inputs=logits,
                                epsilon=1e-5,
                                momentum=0.9,
                                axis=1, 
                                training=True) 
                        logits = self.leaky_relu(logits)
                        logits = tf.layers.dropout(logits, rate=self.keep_prob)
                        # print(logits.get_shape())

                        logits = tf.layers.conv2d(
                                inputs=logits, 
                                filters=256, 
                                kernel_size=5, 
                                strides=2, 
                                padding='same',
                                kernel_initializer=initializer) 
                        logits = tf.layers.batch_normalization(
                                inputs=logits, 
                                momentum=0.9,
                                epsilon=1e-5,
                                axis=1, 
                                training=True)
                        logits = self.leaky_relu(logits)
                        logits = tf.layers.dropout(logits, rate=self.keep_prob)
                        # print(logits.get_shape())

                        # logits = tf.layers.conv2d(
                        #         inputs=logits, 
                        #         filters=512, 
                        #         kernel_size=5, 
                        #         strides=2, 
                        #         padding='same', 
                        #         kernel_initializer=initializer) 
                        # logits = tf.layers.batch_normalization(
                        #         inputs=logits,
                        #         momentum=0.9,
                        #         axis=1, 
                        #         training=True)
                        # logits = self.leaky_relu(logits) 

                        logits = tf.layers.conv2d(
                                inputs=logits, 
                                filters=512, 
                                kernel_size=5, 
                                strides=2, 
                                padding='same', 
                                kernel_initializer=initializer)
                        logits = tf.layers.batch_normalization(
                                inputs=logits, 
                                epsilon=1e-5,
                                momentum=0.9, 
                                axis=1, 
                                training=True)
                        logits = self.leaky_relu(logits)


                        # logits = tf.layers.dropout(logits, rate=self.keep_prob)
                        print(logits.get_shape().as_list())
                        # print(logits.get_shape())

                        logits = tf.layers.flatten(logits)

                        # w = tf.get_variable('w', shape=[logits.get_shape().as_list()[-1], 1], initializer=tf.random_normal_initializer(0.02))
                        # # logits = tf.layers.dense(
                        # #       inputs=logits, 
                        # #       units=1, 
                        # #       activation=tf.nn.sigmoid)
                        # b = tf.get_variable('b', shape=[1], initializer=tf.zeros_initializer())
                        # logits_raw = tf.nn.xw_plus_b(logits, w, b)
                        # print(logits.get_shape())
                        logits_raw = tf.layers.dense(
                                inputs=logits, 
                                units=1)
                        logits = tf.nn.sigmoid(logits_raw)
                        return logits, logits_raw 

class Generator():
        def __init__(self, alpha=0.2, lr=1e-3, beta1=0.5, keep_prob=0.5): 
                self.alpha = alpha
                self.lr = lr
                self.beta1 = beta1
                self.keep_prob = keep_prob

        def leaky_relu(self, x): 
                #return tf.nn.leaky_relu(x, alpha=self.alpha) 
                return tf.maximum(x, self.alpha * x)

        def get_loss(self, logits_real_raw, logits_real, logits_fake_raw, logits_fake):
                G_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake_raw), logits=logits_fake_raw)
                G_loss = tf.reduce_mean(G_loss)
                return G_loss

        def get_train_op(self, G_loss):
                G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
                return tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(G_loss, var_list=G_vars)

        # 16384
        def build_graph(self, z):
                with tf.variable_scope('generator'):
                    return self.common_alg(z, True) 

        def sampler(self, z):
            with tf.variable_scope('generator') as scope: 
                scope.reuse_variables()
                return self.common_alg(z, False)

        def common_alg(self, z, training):
                        initializer = tf.random_normal_initializer(0.02)
                        # w = tf.get_variable('w', shape=[z.get_shape().as_list()[-1], 4*4*512], initializer=tf.random_normal_initializer(0.02))
                        # b = tf.get_variable('b', shape=[4*4*512], initializer=tf.zeros_initializer())
                        # img = tf.nn.xw_plus_b(z, w, b)
                        #
                        # img = tf.reshape(img, (-1, 4, 4, 512))
                        # print(img.get_shape())
                         
                        img = tf.layers.dense(
                                inputs=z, 
                                units=4*4*512) 
                        img = tf.reshape(img, (-1, 4, 4, 512))
                        img = tf.layers.batch_normalization(
                                inputs=img,
                                momentum=0.9,
                                axis=1, 
                                epsilon=1e-5,
                                training=training)
                        img = tf.nn.relu(img)
                        # img = tf.layers.dropout(img, rate=self.keep_prob)

                        # img = tf.layers.conv2d_transpose(
                        #         inputs=img, 
                        #         filters=512, 
                        #         kernel_size=5, 
                        #         strides=1, 
                        #         padding='same', 
                        #         kernel_initializer=initializer)
                        # img = tf.layers.batch_normalization(
                        #         inputs=img, 
                        #         momentum=0.9,
                        #         training=training)
                        # img = tf.nn.relu(img)
                        # img = tf.layers.batch_normalization(
                        #       inputs=img,
                        #       training=True) 
                        # print(img.get_shape())

                        # img = tf.reshape(img, (-1, 4, 4, 512))
                        # img = tf.nn.relu(img) 

                        # img = tf.layers.conv2d_transpose(
                        #         inputs=img, 
                        #         filters=512, 
                        #         kernel_size=5, 
                        #         strides=2, 
                        #         padding='same',
                        #         kernel_initializer=initializer)
                        # img = tf.layers.batch_normalization(
                        #         inputs=img, 
                        #         momentum=0.9,
                        #         axis=1, 
                        #         training=training)
                        # img = tf.nn.relu(img) 
                        
                        # img = tf.layers.conv2d_transpose(
                        #         inputs = img, 
                        #         filters = 1024, 
                        #         kernel_size=5, 
                        #         strides=2, 
                        #         padding='same', 
                        #         kernel_initializer=initializer) 
                        # img = tf.layers.batch_normalization(
                        #         inputs=img, 
                        #         momentum=0.9, 
                        #         axis=1, 
                        #         training=training)
                        # img = tf.nn.relu(img) 

                        img = tf.layers.conv2d_transpose(
                                inputs=img, 
                                filters=256, 
                                kernel_size=5, 
                                strides=2, 
                                padding='same', 
                                kernel_initializer=initializer)
                        img = tf.layers.batch_normalization(
                                inputs=img, 
                                momentum=0.9,
                                axis=1, 
                                epsilon=1e-5,
                                training=training)
                        img = tf.nn.relu(img)
                        img = tf.layers.dropout(img, rate=self.keep_prob)
                        # img = tf.layers.batch_normalization(
                        #       inputs=img, 
                        #       training=True) 
                        # print(img.get_shape())

                        img = tf.layers.conv2d_transpose(
                                inputs=img, 
                                filters=128, 
                                kernel_size=5, 
                                strides=2, 
                                padding='same', 
                                kernel_initializer=initializer)
                        img = tf.layers.batch_normalization(
                                inputs=img, 
                                momentum=0.9,
                                axis=1,
                                epsilon=1e-5,
                                training=training)
                        img = tf.nn.relu(img) 
                        img = tf.layers.dropout(img, rate=self.keep_prob)
                        # img = tf.layers.batch_normalization(
                        #       inputs=img,
                        #       training=True)
                        # print(img.get_shape())

                        img = tf.layers.conv2d_transpose(
                                inputs=img, 
                                filters=64, 
                                kernel_size=5, 
                                strides=2, 
                                padding='same', 
                                kernel_initializer=initializer)
                        img = tf.layers.batch_normalization(
                                inputs=img, 
                                momentum=0.9,
                                axis=1,
                                epsilon=1e-5,
                                training=training)
                        img = tf.nn.relu(img) 
                        img = tf.layers.dropout(img, rate=self.keep_prob) 
                        # img = tf.layers.batch_normalization(
                        #       inputs=img, 
                        #       training=True) 
                        # print(img.get_shape())

                        img = tf.layers.conv2d_transpose(
                                inputs=img, 
                                filters=3, 
                                kernel_size=5, 
                                strides=2, 
                                padding='same',
                                kernel_initializer=initializer)
                        print(img.get_shape().as_list())
                        img = tf.nn.tanh(img)
                        # print(img.get_shape())
                        return img

