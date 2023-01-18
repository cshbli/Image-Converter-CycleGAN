
import os
import tensorflow as tf
from module import discriminator, generator_resnet
from utils import l1_loss, l2_loss, cross_entropy_loss
from datetime import datetime

class CycleGAN(object):

    # def __init__(self, input_size, num_filters = 64, discriminator = discriminator, generator = generator_resnet, lambda_cycle = 10, mode = 'train', log_dir = './log'):
    def __init__(self, input_size, num_filters = 64, discriminator = discriminator, generator = generator_resnet, lambda_cycle = 10, mode = 'train', loss_function='l2', log_dir = './log'):

        self.input_size = input_size

        self.discriminator = discriminator
        self.generator = generator
        self.lambda_cycle = lambda_cycle
        self.num_filters = num_filters
        self.mode = mode
        self.loss_function = loss_function

        self.build_model()
        self.optimizer_initializer()

        # self.saver = tf.train.Saver()
        self.saver = tf.train.Saver(max_to_keep=0)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            self.generator_summaries, self.discriminator_summaries = self.summary()

    def build_model(self):

        # Placeholders for real training samples
        self.input_A_real = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_A_real')
        self.input_B_real = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_B_real')
        # Placeholders for fake generated samples
        self.input_A_fake = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_A_fake')
        self.input_B_fake = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_B_fake')
        # Placeholder for test samples
        self.input_A_test = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_A_test')
        self.input_B_test = tf.placeholder(tf.float32, shape = [None] + self.input_size, name = 'input_B_test')

        self.generation_B = self.generator(inputs = self.input_A_real, num_filters = self.num_filters, reuse = False, scope_name = 'generator_A2B')
        self.cycle_A = self.generator(inputs = self.generation_B, num_filters = self.num_filters, reuse = False, scope_name = 'generator_B2A')

        self.generation_A = self.generator(inputs = self.input_B_real, num_filters = self.num_filters, reuse = True, scope_name = 'generator_B2A')
        self.cycle_B = self.generator(inputs = self.generation_A, num_filters = self.num_filters, reuse = True, scope_name = 'generator_A2B')

        self.discrimination_A_fake = self.discriminator(inputs = self.generation_A, num_filters = self.num_filters, reuse = False, scope_name = 'discriminator_A')
        self.discrimination_B_fake = self.discriminator(inputs = self.generation_B, num_filters = self.num_filters, reuse = False, scope_name = 'discriminator_B')

        # Cycle loss
        # we are able to get the image back using another generator and thus the difference between the original image and the cyclic image should be as small as possible.
        self.cycle_loss = l1_loss(y = self.input_A_real, y_hat = self.cycle_A) + l1_loss(y = self.input_B_real, y_hat = self.cycle_B)

        # Generator loss
        # Generator wants to fool discriminator
        if self.loss_function == 'l2':
            self.generator_loss_A2B = l2_loss(y = tf.ones_like(self.discrimination_B_fake), y_hat = self.discrimination_B_fake)
            self.generator_loss_B2A = l2_loss(y = tf.ones_like(self.discrimination_A_fake), y_hat = self.discrimination_A_fake)
        elif self.loss_function == 'l1':
            self.generator_loss_A2B = l1_loss(y = tf.ones_like(self.discrimination_B_fake), y_hat = self.discrimination_B_fake)
            self.generator_loss_B2A = l1_loss(y = tf.ones_like(self.discrimination_A_fake), y_hat = self.discrimination_A_fake)

        # Merge the two generators and the cycle loss
        # The multiplicative factor of lambda_cycle=10 for cyc_loss assigns more importance to cyclic loss than the discrimination loss
        self.generator_loss = self.generator_loss_A2B + self.generator_loss_B2A + self.lambda_cycle * self.cycle_loss

        # Discriminator output
        self.discrimination_input_A_real = self.discriminator(inputs = self.input_A_real, num_filters = self.num_filters, reuse = True, scope_name = 'discriminator_A')
        self.discrimination_input_B_real = self.discriminator(inputs = self.input_B_real, num_filters = self.num_filters, reuse = True, scope_name = 'discriminator_B')
        self.discrimination_input_A_fake = self.discriminator(inputs = self.input_A_fake, num_filters = self.num_filters, reuse = True, scope_name = 'discriminator_A')
        self.discrimination_input_B_fake = self.discriminator(inputs = self.input_B_fake, num_filters = self.num_filters, reuse = True, scope_name = 'discriminator_B')

        # Discriminator wants to classify real and fake correctly
        # Discriminator must be trained such that recommendation for images from category A must be as close to 1, and vice versa for discriminator B. 
        if self.loss_function == 'l2':
            self.discriminator_loss_input_A_real = l2_loss(y = tf.ones_like(self.discrimination_input_A_real), y_hat = self.discrimination_input_A_real)
            self.discriminator_loss_input_A_fake = l2_loss(y = tf.zeros_like(self.discrimination_input_A_fake), y_hat = self.discrimination_input_A_fake)
        elif self.loss_function == 'l1':
            self.discriminator_loss_input_A_real = l1_loss(y = tf.ones_like(self.discrimination_input_A_real), y_hat = self.discrimination_input_A_real)
            self.discriminator_loss_input_A_fake = l1_loss(y = tf.zeros_like(self.discrimination_input_A_fake), y_hat = self.discrimination_input_A_fake)
        self.discriminator_loss_A = (self.discriminator_loss_input_A_real + self.discriminator_loss_input_A_fake) / 2

        if self.loss_function == 'l2':
            self.discriminator_loss_input_B_real = l2_loss(y = tf.ones_like(self.discrimination_input_B_real), y_hat = self.discrimination_input_B_real)
            self.discriminator_loss_input_B_fake = l2_loss(y = tf.zeros_like(self.discrimination_input_B_fake), y_hat = self.discrimination_input_B_fake)
        elif self.loss_function == 'l1':
            self.discriminator_loss_input_B_real = l1_loss(y = tf.ones_like(self.discrimination_input_B_real), y_hat = self.discrimination_input_B_real)
            self.discriminator_loss_input_B_fake = l1_loss(y = tf.zeros_like(self.discrimination_input_B_fake), y_hat = self.discrimination_input_B_fake)
        self.discriminator_loss_B = (self.discriminator_loss_input_B_real + self.discriminator_loss_input_B_fake) / 2

        # Merge the two discriminators into one
        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B

        # Categorize variables because we have to optimize the two sets of the variables separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]
        #for var in t_vars: print(var.name)

        # Reserved for test
        self.generation_B_test = self.generator(inputs = self.input_A_test, num_filters = self.num_filters, reuse = True, scope_name = 'generator_A2B')
        self.generation_A_test = self.generator(inputs = self.input_B_test, num_filters = self.num_filters, reuse = True, scope_name = 'generator_B2A')


    def optimizer_initializer(self):

        self.learning_rate = tf.placeholder(tf.float32, None, name = 'learning_rate')
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = 0.5).minimize(self.discriminator_loss, var_list = self.discriminator_vars)
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = 0.5).minimize(self.generator_loss, var_list = self.generator_vars) 

    def train(self, input_A, input_B, learning_rate):

        generation_A, generation_B, generator_loss, _, generator_summaries = self.sess.run(
            [self.generation_A, self.generation_B, self.generator_loss, self.generator_optimizer, self.generator_summaries], \
            feed_dict = {self.input_A_real: input_A, self.input_B_real: input_B, self.learning_rate: learning_rate})

        self.writer.add_summary(generator_summaries, self.train_step)

        discriminator_loss, _, discriminator_summaries = self.sess.run([self.discriminator_loss, self.discriminator_optimizer, self.discriminator_summaries], \
            feed_dict = {self.input_A_real: input_A, self.input_B_real: input_B, self.learning_rate: learning_rate, self.input_A_fake: generation_A, self.input_B_fake: generation_B})

        self.writer.add_summary(discriminator_summaries, self.train_step)

        self.train_step += 1

        return generator_loss, discriminator_loss


    def test(self, inputs, direction):

        if direction == 'A2B':
            generation = self.sess.run(self.generation_B_test, feed_dict = {self.input_A_test: inputs})
        elif direction == 'B2A':
            generation = self.sess.run(self.generation_A_test, feed_dict = {self.input_B_test: inputs})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        
        return os.path.join(directory, filename)

    def load(self, filepath):
        checkpoint = tf.train.latest_checkpoint(filepath)
        #self.saver.restore(self.sess, filepath)
        self.saver.restore(self.sess, checkpoint)

    def summary(self):

        with tf.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', self.cycle_loss)
            generator_loss_A2B_summary = tf.summary.scalar('generator_loss_A2B', self.generator_loss_A2B)
            generator_loss_B2A_summary = tf.summary.scalar('generator_loss_B2A', self.generator_loss_B2A)
            generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries = tf.summary.merge([cycle_loss_summary, generator_loss_A2B_summary, generator_loss_B2A_summary, generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_A_summary = tf.summary.scalar('discriminator_loss_A', self.discriminator_loss_A)
            discriminator_loss_B_summary = tf.summary.scalar('discriminator_loss_B', self.discriminator_loss_B)
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
            discriminator_summaries = tf.summary.merge([discriminator_loss_A_summary, discriminator_loss_B_summary, discriminator_loss_summary])

        return generator_summaries, discriminator_summaries


