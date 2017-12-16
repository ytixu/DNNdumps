import numpy as np
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, Input, LeakyReLU, Activation, BatchNormalization
from keras.optimizers import Adam, SGD

from utils import arg_parse, image
from _gan_model import gan_model

class GAN(gan_model, object):

	def __init__(self, args):
		self.D = None   # discriminator
		self.G = None   # generator
		self.AM = None  # adversarial model
		self.DM = None  # discriminator model
		super(GAN, self).__init__(args)

	def discriminator(self):
		if self.D:
			return self.D

		self.D = Sequential()

		self.D.add(Flatten(input_shape=self.input_shape))
		self.D.add(Dense(self.input_shape[0]))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dense(self.input_shape[0]/2))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dense(1, activation='sigmoid'))
		self.D.summary()

		return self.D

	def generator(self):
		if self.G:
			return self.G

		self.G = Sequential()

		self.G.add(Flatten(input_shape=self.latent_shape))
		self.G.add(Dense(self.input_shape[0]))
		self.G.add(LeakyReLU(alpha=0.2))
		self.G.add(BatchNormalization(momentum=0.8))
		self.G.add(Dense(self.input_shape[0]*2, activation='tanh'))
		self.G.add(Reshape(self.input_shape))

		self.G.summary()
		return self.G


	def discriminator_model(self):
		if self.DM:
			return self.DM
		optimizer = SGD(lr=0.001, decay=3e-8)
		self.DM = Sequential()
		self.DM.add(self.discriminator())
		self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.DM

	def adversarial_model(self):
		if self.AM:
			return self.AM
		optimizer = Adam(lr=0.0001, decay=3e-8)
		self.AM = Sequential()
		self.AM.add(self.generator())
		self.AM.add(self.discriminator())
		self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.AM

	def sample_Z(self):
		return self.Z[np.random.choice(range(len(self.Z)), self.batch_size),:]

	def sample(self):
		z = self.sample_Z()
		return self.generator().predict(z)

	def fit(self, x_data, z_data):
		self.set_data(x_data, z_data)

		for i in range(self.epoch):
			true_x = self.X[np.random.choice(range(len(self.X)), self.batch_size),:]
			fake_x = self.sample()

			x = np.concatenate((true_x, fake_x))
			y = np.ones([2*self.batch_size, 1])
			y[self.batch_size:, :] = 0
			# fake ones are 1

			d_loss = self.discriminator_model().train_on_batch(x, y)
			
			y = np.ones([self.batch_size, 1])
			z = self.sample_Z()
			a_loss = self.adversarial_model().train_on_batch(z, y)

			if i == 0:			
				image.plot_batch(true_x, self.batch_size)

			if i % 1000 == 1:
				log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
				log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
				print(log_mesg)

				image.generate_and_plot(self)

	


if __name__ == '__main__':
	x, y, config = arg_parse.get_parse()
	# print x
	gan = GAN(config)
	gan.start_timer()
	gan.fit(x, y)
	gan.elapsed_time()
	# image.generate_and_plot(gan)