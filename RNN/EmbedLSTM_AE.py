import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

from utils import parser, image, embedding_plotter
from LSTM_AE import LSTM_AE

class Embedded_LSTM_AE:

	def __init__(self, args):
		self.model = None
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

		self.trained = args['mode'] == 'sample' if 'mode' in args else False
		self.load_path = args['load_path']

		args['load_path'] = args['embedded_model_load_path']
		self.autoencoder = LSTM_AE(args)
		self.autoencoder.load()

	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.input_dim))
		encoded = LSTM(self.latent_dim)(inputs)

		z = Input(shape=(self.latent_dim, ))
		decode_1 = RepeatVector(self.timesteps)
		decode_2 = LSTM(self.output_dim, return_sequences=True)
		
		decoded = decode_1(encoded)
		decoded = decode_2(decoded)
		
		decoded_ = decode_1(z)
		decoded_ = decode_2(decoded_)
		
		self.encoder = Model(inputs, encoded)
		self.encoder.compile(optimizer='RMSprop', loss='mean_squared_error')

		self.decoder = Model(z, decoded_)
		self.decoder.compile(optimizer='RMSprop', loss='mean_squared_error')

		self.model = Model(inputs, decoded)
		
		self.model.summary()
		self.encoder.summary()
		self.decoder.summary()

	def run(self, data_iterator): 
		self.make_model()

		if self.trained:
			self.model.load_weights(self.load_path)
			embedding_plotter.see_embedding(self.encoder, data_iterator)

		else:
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
					
					data_train = np.concatenate((x_train,y_train), axis=2)
					data_test = np.concatenate((x_test,y_test), axis=2)
					
					e_train = self.autoencoder.encoder.predict(data_train)
					e_test = self.autoencoder.encoder.predict(data_test)
				
					self.encoder.fit(x_train, e_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=(x_test, e_test))
					self.decoder.fit(e_train, y_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=(e_test, y_test))

					y_test_decoded = self.model.predict(x_test[:1])
					image.plot_batch_1D(y_test[:1], y_test_decoded)
					self.model.save_weights(self.load_path, overwrite=True)

				iter1, iter2 = tee(iter2)

			embedding_plotter.see_embedding(self.encoder, iter2)

if __name__ == '__main__':
	data_iterator, config = parser.get_parse('Embedded_LSTM_AE')
	config['embedded_model_load_path'] = config['load_path']
	ae = Embedded_LSTM_AE(config)
	ae.run(data_iterator)
