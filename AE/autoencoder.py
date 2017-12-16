import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics, objectives

from _ae_model import ae_model
from DNNdumps.utils import arg_parse, image

class AE(ae_model, object):

	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		self.decoder = None
		super(AE, self).__init__(args)

	
	def make_model(self):
		x_input = Input(shape=(self.input_dim,))
		h = Dense(self.input_intermediate_dim, activation='relu')(x_input)
		z_mean = Dense(self.latent_dim)(h)
		z_log_var = Dense(self.latent_dim)(h)

		def sampling(args):
			z_mean, z_log_var = args
			epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=self.epsilon_std)
			return z_mean + K.exp(z_log_var / 2) * epsilon

		# note that "output_shape" isn't necessary with the TensorFlow backend
		z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

		# we instantiate these layers separately so as to reuse them later
		decoder_h = Dense(self.output_intermediate_dim, activation='relu')
		decoder_mean = Dense(self.output_dim, activation='tanh')
		h_decoded = decoder_h(z)
		y_decoded_mean = decoder_mean(h_decoded)

		# build a model to project inputs on the latent space
		self.encoder = Model(x_input, z_mean)
		self.encoder.summary()

		# build a decoder that can sample from the learned distribution
		decoder_input = Input(shape=(self.latent_dim,))
		_h_decoded = decoder_h(decoder_input)
		_y_decoded_mean = decoder_mean(_h_decoded)
		self.decoder = Model(decoder_input, _y_decoded_mean)
		self.decoder.summary()

		# y = CustomVariationalLayer()([x_input, y_decoded_mean])
		self.autoencoder = Model(x_input, y_decoded_mean)
		self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
		self.autoencoder.summary()

		
	def load_model(self):
		self.autoencoder.load_weights(self.load_path)
		# self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

	def interpolate(self, x, n=6):
		poses = [None]*((n+1)**2)
		for i in range(n+1):
			x1, x2, = x[np.random.randint(len(x))], x[np.random.randint(len(x))]

			for j in range(n):
				diff = (x2 - x1)/n
				poses[i*(n+1)+j] = x1+diff*j
			poses[i*(n+1)+n] = x2
		
		poses = self.autoencoder.predict(np.array(poses))
		image.plot_batch(poses, (n+1)**2)

	def run(self, x, y):
		self.set_input_dim(x, y)
		self.make_model()

		if self.mode == 'sample':
			self.load_model()
			scores = self.autoencoder.evaluate(x, y, verbose=0)
			print'acc:', scores
	
			self.interpolate(x)

		else:
			n = range(len(x))

			for i in range(100):
				l = np.random.choice(n, self.batch_size/2)
				_l = list(set(n) - set(l))
				x_test = x[l]
				x_train = x[_l]
				y_test = y[l]
				y_train = y[_l]
				self.autoencoder.fit(x_train, y_train,
						shuffle=True,
						epochs=self.epochs,
						batch_size=self.batch_size,
						validation_data=(x_test, y_test))

				# display a 2D plot of the digit classes in the latent space
				z_test_encoded = self.encoder.predict(x_test[:8])
				y_test_decoded = self.decoder.predict(z_test_encoded)
				image.plot_batch(np.concatenate([y_test[:8], y_test_decoded], axis=0), 16)
				# image.plot_batch(np.concatenate([x_test, x_test], axis=0), self.batch_size)
				self.autoencoder.save_weights(self.load_path, overwrite=True)




if __name__ == '__main__':
	x, y, config = arg_parse.get_parse()
	ae = AE(config)
	ae.start_timer()
	ae.run(x, y)
	ae.elapsed_time()
