import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, RepeatVector, Lambda, concatenate
from keras.models import Model

from utils import parser, image, embedding_plotter, recorder, metrics

NAME = 'H_LSTM'
USE_GRU = False
if USE_GRU:
	from keras.layers import GRU
else:
	from keras.layers import LSTM

class H_LSTM:
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
		# self.hierarchies = args['hierarchies'] if 'hierarchies' in args else 4
		self.latent_dim = args['latent_dim'] if 'latent_dim' in args else (args['input_dim']+args['output_dim'])/2
		self.trained = args['mode'] == 'sample' if 'mode' in args else False
		self.load_path = args['load_path']
		self.log_path = args['log_path']

		self.history = recorder.LossHistory()

	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.input_dim))
		encoded = None
		if USE_GRU:
			encoded = GRU(self.latent_dim, return_sequences=True)(inputs)
		else:		
			encoded = LSTM(self.latent_dim, return_sequences=True)(inputs)

		z = Input(shape=(self.latent_dim,))
		decode_1 = RepeatVector(self.timesteps)
		decoder_2 = None 
		if USE_GRU:
			decode_2 = GRU(self.output_dim, return_sequences=True)
		else:
			decode_2 = LSTM(self.output_dim, return_sequences=True)

		decoded = [None]*self.timesteps
		for i in range(self.timesteps):
			e = Lambda(lambda x: x[:,i], output_shape=(self.latent_dim,))(encoded)
			decoded[i] = decode_1(e)
			decoded[i] = decode_2(decoded[i])
		decoded = concatenate(decoded, axis=1)

		decoded_ = decode_1(z)
		decoded_ = decode_2(decoded_)
		
		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.autoencoder = Model(inputs, decoded)
		self.autoencoder.compile(optimizer='RMSprop', loss='mean_squared_error')

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
		return np.reshape(y, (-1, self.timesteps**2, y.shape[-1]))
		

	def run(self, data_iterator, valid_data): 
		model_vars = [NAME, self.latent_dim, self.timesteps, self.batch_size]
		if not self.load():
			# from keras.utils import plot_model
			# plot_model(self.autoencoder, to_file='model.png')
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
					y_train = self.__alter_y(y_train)
					y_test_orig = np.copy(y_test[:1])
					y_test = self.__alter_y(y_test)
					history = self.autoencoder.fit(x_train, y_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=(x_test, y_test),
								callbacks=[self.history])

					y_test_decoded = self.autoencoder.predict(x_test[:1])
					image.plot_hierarchies(y_test_orig, y_test_decoded)
					self.autoencoder.save_weights(self.load_path, overwrite=True)
				iter1, iter2 = tee(iter2)
			
			self.history.record(self.log_path, model_vars)
			data_iterator = iter2
		embedding_plotter.see_hierarchical_embedding(self.encoder, self.decoder, data_iterator, valid_data, model_vars)
		# iter1, iter2 = tee(data_iterator)
		# metrics.validate(valid_data, self.encoder, self.decoder, self.timesteps, metrics.H_LSTM)
		# evaluate.eval_pattern_reconstruction(self.encoder, self.decoder, iter2)

if __name__ == '__main__':
	data_iterator, valid_data, config = parser.get_parse(NAME)
	ae = H_LSTM(config)
	ae.run(data_iterator, valid_data)
