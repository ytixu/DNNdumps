import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.optimizers import Adam

class ElapsedTimer(object):
	def __init__(self):
		self.start_time = time.time()
	
	def elapsed(self,sec):
		if sec < 60:
			return str(sec) + " sec"
		elif sec < (60 * 60):
			return str(sec / 60) + " min"
		else:
			return str(sec / (60 * 60)) + " hr"

	def elapsed_time(self):
		print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )


class GAN:

	def __init__(self, batch_size):
		self.D = None   # discriminator
		self.G = None   # generator
		self.AM = None  # adversarial model
		self.DM = None  # discriminator model

		self.batch_size = batch_size
		self.input_size = 0
		self.latent_size = 0
		self.X = None

	def discriminator(self):
		if self.D:
			return self.D

		self.D = Sequential([
					Dense(self.input_size/2, input_shape=(self.input_size,), activation='relu'),
					Dense(self.input_size/4, activation='relu'),
					Dense(1, activation='sigmoid')
				])

		self.D.summary()
		return self.D

	def generator(self):
		if self.G:
			return self.G

		self.G = Sequential([
					Dense(self.input_size/2, input_shape=(self.latent_size,), activation='relu'),
					Dense(self.input_size, activation='tanh')
				])

		self.G.summary()
		return self.G


	def discriminator_model(self):
		if self.DM:
			return self.DM
		optimizer = Adam(lr=0.0002, decay=6e-8)
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

	def load_data(self, data=[], file_name=None):
		if data.any():
			self.X = data
		elif file_name:
			self.X = np.array([map(float, x.split()) for x in open(file_name, 'r').read().split('\n')])/self.norm_max
		else:
			return

		self.input_size = len(self.X[0])
		self.latent_size = self.input_size/4
		self.norm_max = np.amax(np.amax(np.abs(self.X)))


	def train(self, data, train_steps=5):
		self.load_data(data)

		for i in range(train_steps):
			true_x = self.X[np.random.choice(range(len(self.X)), self.batch_size),:]
			noise_x = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.latent_size])
			fake_x = self.generator().predict(noise_x)

			x = np.concatenate((true_x, fake_x))
			y = np.ones([2*self.batch_size, 1])
			y[self.batch_size:, :] = 0
			# fake ones are 1

			d_loss = self.discriminator_model().train_on_batch(x, y)
			
			y = np.ones([self.batch_size, 1])
			noise_x = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.latent_size])
			a_loss = self.adversarial_model().train_on_batch(noise_x, y)

			log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
			log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
			print(log_mesg)

	def plot_image(self):
		noise_x = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.latent_size])
		fake_x = self.generator().predict(noise_x)
		
		for x in fake_x:
			xs = [x[i]*self.norm_max for i in range(0, len(x), 3)]
			ys = [x[i]*self.norm_max for i in range(1, len(x), 3)]
			zs = [x[i]*self.norm_max for i in range(2, len(x), 3)]

			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.plot(xs, ys, zs)
			plt.show()


def random_data(n=50, b=16, j=6):
	x = np.zeros((n*b, j*3))
	for i in range(j*3):
		coord = np.random.randint(1,9)
		x[:,i] = np.random.normal(coord, 0.1, n*b)
		print i, coord

	return x

if __name__ == '__main__':
	gan = GAN(16)
	x = random_data()
	# print x
	timer = ElapsedTimer()
	gan.train(x)
	timer.elapsed_time()
	gan.plot_image()
