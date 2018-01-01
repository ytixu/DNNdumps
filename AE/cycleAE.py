import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics, objectives

from _ae_model import ae_model
from DNNdumps.utils import arg_parse, image

class cycleAE(ae_model, object):

	def __init__(self, args):
		self.autoencoder = {}
		self.encoder = {}
		self.decoder = {}
		self.cycleAE = {}
		
		super(cycleAE, self).__init__(args)

	def set_input_dim(self, x, y):
		self.input_dim = [len(x[0]), len(y[0])]
		self.output_dim = [len(y[0]), len(x[0])]
		self.latent_dim = (self.input_dim[0]+self.output_dim[0])/2
		self.input_intermediate_dim = [(self.input_dim[i]+self.latent_dim)/2 for i in range(2)]
		self.output_intermediate_dim = [(self.output_dim[i]+self.latent_dim)/2 for i in range(2)]

	def make_model(self):
		x_inputs = [None]*2
		for cycle_n in range(2):
			x_inputs[cycle_n] = Input(shape=(self.input_dim[cycle_n],))
			h = Dense(self.input_intermediate_dim[cycle_n], activation='relu')(x_inputs[cycle_n])
			z_mean = Dense(self.latent_dim)(h)
			z_log_var = Dense(self.latent_dim)(h)
			
			def sampling(args):
				z_mean, z_log_var = args
				epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=self.epsilon_std)
				return z_mean + K.exp(z_log_var / 2) * epsilon

			# note that "output_shape" isn't necessary with the TensorFlow backend
			z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

			# we instantiate these layers separately so as to reuse them later
			decoder_h = Dense(self.output_intermediate_dim[cycle_n], activation='relu')
			decoder_mean = Dense(self.output_dim[cycle_n], activation='tanh')
			h_decoded = decoder_h(z)
			y_decoded_mean = decoder_mean(h_decoded)

			# build a model to project inputs on the latent space
			self.encoder[cycle_n] = Model(x_inputs[cycle_n], z_mean)
			self.encoder[cycle_n].summary()

			# build a decoder that can sample from the learned distribution
			decoder_input = Input(shape=(self.latent_dim,))
			_h_decoded = decoder_h(decoder_input)
			_y_decoded_mean = decoder_mean(_h_decoded)
			self.decoder[cycle_n] = Model(decoder_input, _y_decoded_mean)
			self.decoder[cycle_n].summary()

			# y = CustomVariationalLayer()([x_inputs[cycle_n], y_decoded_mean])
			self.autoencoder[cycle_n] = Model(x_inputs[cycle_n], y_decoded_mean)
			self.autoencoder[cycle_n].compile(optimizer='adadelta', loss='mean_squared_error')
			self.autoencoder[cycle_n].summary()

		for cycle_n in range(2):
			x = self.autoencoder[cycle_n](x_inputs[cycle_n])
			predictions = self.autoencoder[(cycle_n+1)%2](x)
			self.cycleAE[cycle_n] = Model(x_inputs[cycle_n], predictions)
			self.cycleAE[cycle_n].compile(optimizer='adadelta', loss='mean_squared_error')


	# def load_model(self):
	# 	self.autoencoder.load_weights(self.load_path)
	# 	# self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

	# def interpolate(self, x, n=6):
	# 	poses = [None]*((n+1)**2)
	# 	for i in range(n+1):
	# 		x1, x2, = x[np.random.randint(len(x))], x[np.random.randint(len(x))]

	# 		for j in range(n):
	# 			diff = (x2 - x1)/n
	# 			poses[i*(n+1)+j] = x1+diff*j
	# 		poses[i*(n+1)+n] = x2
		
	# 	poses = self.autoencoder.predict(np.array(poses))
	# 	image.plot_batch(poses, (n+1)**2)

	def run(self, x, y):
		self.set_input_dim(x, y)
		self.make_model()

		if self.mode == 'sample':
			self.load_model()
			scores = self.autoencoder[0].evaluate(x, y, verbose=0)
			print'acc:', scores
	
			# self.interpolate(x)

		else:
			n = range(len(x))

			for i in range(100):
				l = np.random.choice(n, self.batch_size/2)
				_l = list(set(n) - set(l))
				test = [x[l], y[l]]
				dev = [x[_l], y[_l]]
				for cycle_n in range(2):
					self.autoencoder[cycle_n].fit(dev[cycle_n], dev[(cycle_n+1)%2],
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							validation_data=(test[cycle_n], test[(cycle_n+1)%2]))

				# display a 2D plot of the digit classes in the latent space
					z_test_encoded = self.encoder[cycle_n].predict(test[cycle_n][:8])
					y_test_decoded = self.decoder[cycle_n].predict(z_test_encoded)
					image.plot_batch(np.concatenate([test[(cycle_n+1)%2][:8], y_test_decoded], axis=0), 16)
				# image.plot_batch(np.concatenate([x_test, x_test], axis=0), self.batch_size)
				# self.autoencoder.save_weights(self.load_path, overwrite=True)




if __name__ == '__main__':
	x, y, config = arg_parse.get_parse()
	ae = cycleAE(config)
	ae.start_timer()
	ae.run(x, y)
	ae.elapsed_time()
