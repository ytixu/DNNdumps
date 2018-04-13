import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, LSTM, RepeatVector, Flatten, Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K

from utils import parser, image, embedding_plotter, recorder, option_visualizer


NAME = 'Prior_LSTM'
USE_GRU = True
if USE_GRU:
	from keras.layers import GRU


class Prior_LSTM:

	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		# self.decoder = None

		self.epochs = args['epochs']
		self.batch_size = args['batch_size']
		self.periods = args['periods'] if 'periods' in args else 15
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 5
		self.input_dim = args['input_dim']
		self.output_dim = args['output_dim']
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

		prior_input = Input(shape=(self.output_dim,))
		decoded = concatenate([encoded, prior_input], axis=1)
		decoded = RepeatVector(self.timesteps)(decoded)

		if USE_GRU:
			decoded = GRU(self.output_dim, return_sequences=True)(decoded)
		else:
			decoded = LSTM(self.output_dim, return_sequences=True)(decoded)

		self.encoder = Model(inputs, encoded)
		self.autoencoder = Model([inputs, prior_input], decoded)
		self.autoencoder.compile(optimizer='RMSprop', loss='mean_squared_error')

		self.autoencoder.summary()
		self.encoder.summary()

	def load(self):
		self.make_model()
		if self.trained:
			self.autoencoder.load_weights(self.load_path)
			return True
		return False

	def run(self, data_iterator): 
		if not self.load():
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
					self.autoencoder.fit([x_train, y_train[:,0]], y_train,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							validation_data=([x_test, y_test[:,0]], y_test))
							# callbacks=[self.history])

				y_test_decoded = self.autoencoder.predict([x_test[:1],y_test[:1,0]])
				image.plot_batch_1D(y_test[:1], y_test_decoded)
				self.autoencoder.save_weights(self.load_path, overwrite=True)
				iter1, iter2 = tee(iter2)
			
			data_iterator = iter2

		model_vars = [NAME, self.latent_dim, self.timesteps, self.batch_size]
		embedding_plotter.see_embedding(self.encoder, data_iterator, model_vars)
		# self.history.record(self.log_path, model_vars)

if __name__ == '__main__':
	data_iterator, config = parser.get_parse(NAME)
	ae = Prior_LSTM(config)
	ae.run(data_iterator)
