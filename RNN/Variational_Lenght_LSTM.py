import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, RepeatVector, Lambda, concatenate
from keras.models import Model

from utils import parser, image, embedding_plotter, recorder

NAME = 'VL_LSTM'
USE_GRU = True
if USE_GRU:
	from keras.layers import GRU
else:
	from keras.layers import LSTM

class VL_LSTM:
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
			encoded = GRU(self.latent_dim)(inputs)
		else:
			encoded = LSTM(self.latent_dim)(inputs)

		z = Input(shape=(self.latent_dim,))
		decode_1 = RepeatVector(self.timesteps)
		decode_2 = None
		if USE_GRU:
			decode_2 = GRU(self.output_dim, return_sequences=True)
		else:
			decode_2 = LSTM(self.output_dim, return_sequences=True)

		decoded = decode_1(encoded)
		decoded = decode_2(decoded)

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

	def __alter(self, data):
		inc = data[0].shape[-1]/self.timesteps
		for b in range(self.timesteps):
			new_data = [[]]*4
			for i,x in enumerate(data[b*inc:(b+1)*inc]):
				new_data[i] = np.copy(x)
				new_data[i] = np.repeat(new_data[i], self.timesteps, axis=0)
				new_data[i] = np.reshape(new_data[i], (-1, self.timesteps, self.timesteps, x.shape[-1]))
				for j in range(self.timesteps-1):
					new_data[i][:,j,j+1:,:] = 0.0
				new_data[i] = np.reshape(new_data[i], (-1, self.timesteps, x.shape[-1]))
	
			if len(new_data[0]) > 0:
				yield (new_data[j] for j in range(4))

	def run(self, data_iterator, validation_data): 
		model_vars = [NAME, self.latent_dim, self.timesteps, self.batch_size]
		if not self.load():
			# from keras.utils import plot_model
			# plot_model(self.autoencoder, to_file='model.png')
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					data = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
					for x_train, x_test, y_train, y_test in self.__alter(data):
						print x_train.shape
						self.autoencoder.fit(x_train, y_train,
									shuffle=True,
									epochs=self.epochs,
									batch_size=self.batch_size,
									validation_data=(x_test, y_test),
									callbacks=[self.history])

						y_test_decoded = self.autoencoder.predict(x_test[:self.timesteps])
						y_test_decoded = np.reshape(y_test_decoded, (1, self.timesteps**2, -1))
						image.plot_hierarchies(x_test[self.timesteps-1:self.timesteps], y_test_decoded)
					self.autoencoder.save_weights(self.load_path, overwrite=True)
				iter1, iter2 = tee(iter2)
			
			data_iterator = iter2

			self.history.record(self.log_path, model_vars)

		embedding_plotter.see_variational_length_embedding(self.encoder, self.decoder, data_iterator, validation_data, self.timesteps, model_vars)

if __name__ == '__main__':
	data_iterator, validation_data, config = parser.get_parse(NAME)
	ae = VL_LSTM(config)
	ae.run(data_iterator, validation_data)
