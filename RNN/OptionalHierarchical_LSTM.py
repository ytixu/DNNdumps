import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, RepeatVector, Lambda, concatenate, Reshape, Flatten
from keras.models import Model
from keras import backend as K

from utils import parser, image, embedding_plotter, recorder

NAME = 'OH_LSTM'
# supports:
# 1) Multiple modalities of output (options)
# 2) Distribution vector of most common modalities
# 3) Multiple output sequence length (hierarchies)

USE_GRU = False
if USE_GRU:
	from keras.layers import GRU
else:
	from keras.layers import LSTM

class OH_LSTM:
	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		self.decoder = None

		self.epochs = args['epochs']
		self.batch_size = args['batch_size']
		self.periods = args['periods'] if 'periods' in args else 10
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 5
		self.input_dim = args['input_dim']
		self.output_dim = args['output_dim']
		self.latent_dim = args['latent_dim'] if 'latent_dim' in args else (args['input_dim']+args['output_dim'])/2
		self.option_dim = args['option_dim'] if 'option_dim' in args else 5
		self.out_dim = self.output_dim*self.timesteps
		self.trained = args['mode'] == 'sample' if 'mode' in args else False
		self.load_path = args['load_path']
		self.log_path = args['log_path']

		self.history = recorder.LossHistory()

	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.input_dim))
		encoded = LSTM(self.latent_dim, return_sequences=True)(inputs)

		z = Input(shape=(self.latent_dim,))
		option_1 = RepeatVector(self.option_dim)
		option_2 = Flatten()
		decode_1 = RepeatVector(self.timesteps)
		decode_2 = LSTM(self.output_dim*self.option_dim, return_sequences=True)

		decoded = [None]*self.timesteps
		for i in range(self.timesteps):
			e = Lambda(lambda x: x[:,i], output_shape=(self.latent_dim,))(encoded)
			decoded[i] = option_1(e)
			decoded[i] = option_2(decoded[i])
			decoded[i] = decode_1(decoded[i])
			decoded[i] = decode_2(decoded[i])

		decoded = concatenate(decoded, axis=1)
		decoded = Reshape((self.timesteps, self.option_dim, self.out_dim))(decoded)

		decoded_ = option_1(z)
		decoded_ = option_2(decoded_)
		decoded_ = decode_1(decoded_)
		decoded_ = decode_2(decoded_)
		decoded_ = Reshape((self.option_dim, self.out_dim))(decoded_)
		
		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.autoencoder = Model(inputs, decoded)

		def min_loss(y_true, y_pred):
			b = K.int_shape(y_pred)[0]
			loss_val = None
			for i in range(self.timesteps):
				hry_pred = y_pred[:,i]
				hry_true = K.tile(y_true[:,i], (1, self.option_dim, 1))
				hry_true = K.reshape(hry_true, [-1, self.option_dim, self.out_dim])
				sum_sqr_ = K.sum(K.square(hry_pred - hry_true), axis=2)
				min_ = K.min(sum_sqr_, axis=1)
				if loss_val is None:
					loss_val = min_
				else:
					loss_val += min_
			return loss_val/self.timesteps

		self.autoencoder.compile(optimizer='RMSprop', loss=min_loss)

		self.autoencoder.summary()
		self.encoder.summary()
		self.decoder.summary()

	def load(self):
		self.make_model()
		if self.trained:
			self.autoencoder.load_weights(self.load_path)
			return True
		return False

	def __alter_y(self, y):
		y = np.repeat(y, self.timesteps, axis=0)
		y = np.reshape(y, (-1, self.timesteps, self.timesteps, y.shape[-1]))
		for i in range(self.timesteps-1):
			y[:,i,i+1:,:] = 0.0
		return y
		# return np.reshape(y, (-1, self.timesteps*self.timesteps, y.shape[-1]))
		

	def run(self, data_iterator): 
		model_vars = [NAME, self.latent_dim, self.timesteps, self.batch_size]
		if not self.load():
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
					y_train = self.__alter_y(y_train)
					y_test_orig = np.copy(y_test[:1])
					y_test = self.__alter_y(y_test)
					self.autoencoder.fit(x_train, y_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=(x_test, y_test),
								callbacks=[self.history])

					y_test_decoded = self.autoencoder.predict(x_test[:1])
					image.plot_options_hierarchies(y_test_orig, y_test_decoded)
					# self.autoencoder.save_weights(self.load_path, overwrite=True)
				iter1, iter2 = tee(iter2)
			
			data_iterator = iter2

			# self.history.record(self.log_path, model_vars)

		# embedding_plotter.see_hierarchical_embedding(self.encoder, self.decoder, data_iterator, model_vars)

if __name__ == '__main__':
	data_iterator, config = parser.get_parse(NAME)
	ae = OH_LSTM(config)
	ae.run(data_iterator)
