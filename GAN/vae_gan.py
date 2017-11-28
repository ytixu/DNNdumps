import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.optimizers import Adam

from gan import GAN, ElapsedTimer, random_data

class VAE_GAN(GAN):

	# generator is decoder
	def __init__(self, batch_size):
		self.E = None
		self.EM = None
		super().__init__(batch_size)

	def encoder(self):
		if self.E:
			return self.E

		self.E = Sequential([
					Dense(self.input_size/2, input_shape=(self.input_size,), activation='relu'),
					Dense(self.latent_size, activation='tanh')
				])

		self.E.summary()
		return self.E

	def encoder_model(self):
		if self.EM:
			return self.EM
		optimizer = Adam(lr=0.0001, decay=3e-8)
		self.EM = Sequential()
		self.EM.add(self.encoder())
		self.EM.add(self.generator())
		self.EM.add(self.discriminator())
		self.EM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.EM

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
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			a_loss = self.adversarial_model().train_on_batch(noise, y)

			e_loss = self.encoder_model().train_on_batch(true_x, y)

			log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
			log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
			log_mesg = "%s  [E loss: %f, acc: %f]" % (log_mesg, e_loss[0], e_loss[1])
			print(log_mesg)

if __name__ == '__main__':
	gan = VAE_GAN(16)
	x = random_data()
	# print x
	timer = ElapsedTimer()
	gan.train(x)
	timer.elapsed_time()
	gan.plot_image()
