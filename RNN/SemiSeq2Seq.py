import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

from utils import parser, image

class LSTM_AE:

	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		self.decoder = None

		self.epochs = args['epochs'] if 'epochs' in args else 10
		self.periods = args['periods'] if 'periods' in args else 5
		self.batch_size = args['batch_size'] if 'batch_size' in args else 16
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 5
		self.input_dim = args['input_dim']
		self.output_dim = args['output_dim']
		self.latent_dim = args['latent_dim'] if 'latent_dim' in args else (args['input_dim']+args['output_dim'])/2
		self.trained = args['mode'] == 'sample' if 'mode' in args else False


	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.input_dim))
		encoded = LSTM(self.latent_dim)(inputs)
		self.encoder = Model(inputs, encoded)

		decoded = RepeatVector(self.timesteps)(encoded)
		decoded = LSTM(self.input_dim, return_sequences=True)(decoded)

		inputs_ = Input(shape=(self.latent_dim,))
		decoded_ = RepeatVector(self.timesteps)(inputs_)
		decoded_ = LSTM(self.output_dim, return_sequences=True)(decoded_)
		
		self.encoder = Model(inputs, encoded)
		self.autoencoder = Model(inputs, decoded)
		self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
		self.decoder = Model(inputs_, decoded_)
		self.decoder.compile(optimizer='adadelta', loss='mean_squared_error')

		self.autoencoder.summary()
		self.encoder.summary()
		self.decoder.summary()

	def run(self, data_iterator): 
		self.make_model()

		if self.trained:
			self.autoencoder.load_weights(self.load_path)

		else:
			iter1, iter2 = tee(data_iterator)
			# train on human
			for x, _ in iter1:
				for i in range(self.periods):
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, x, test_size=self.cv_splits)
					self.autoencoder.fit(x_train, y_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=(x_test, y_test))

					y_test_decoded = self.autoencoder.predict(x_test[:1])
					image.plot_batch_1D(y_test[:1], y_test_decoded)
					# self.autoencoder.save_weights(self.load_path, overwrite=True)

			# train on robot
			for x, y in iter2:
				norm_y = y/np.pi/2
				for i in range(self.periods):
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, norm_y, test_size=self.cv_splits)
					x_train_encoded = self.encoder.predict(x_train)
					x_test_encoded = self.encoder.predict(x_test)

					self.decoder.fit(x_train_encoded, y_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=(x_test_encoded, y_test))

					y_test_decoded = self.decoder.predict(x_test_encoded[:1])
					image.plot_batch_1D(y_test[:1], y_test_decoded)
					# self.autoencoder.save_weights(self.load_path, overwrite=True)

if __name__ == '__main__':
	data_iterator, config = parser.get_parse('LSTM_AE')
	ae = LSTM_AE(config)
	ae.run(data_iterator)
