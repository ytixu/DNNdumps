import numpy as np
from itertools import tee
import itertools.product
from sklearn import cross_validation
from keras.layers import Input, LSTM, RepeatVector, Flatten
from keras.models import Model
from keras import backend as K

from utils import parser, image, embedding_plotter, recorder


NAME = 'Option_LSTM'

class Option_LSTM:

	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		# self.decoder = None

		self.epochs = args['epochs']
		self.batch_size = args['batch_size']
		self.periods = args['periods'] if 'periods' in args else 10
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 5
		self.input_dim = args['input_dim']
		self.output_dim = args['output_dim']
		self.latent_dim = args['latent_dim'] if 'latent_dim' in args else (args['input_dim']+args['output_dim'])/2
		self.option_dim = args['option_dim'] if 'option_dim' in args else 2
		self.trained = args['mode'] == 'sample' if 'mode' in args else False
		self.load_path = args['load_path']
		self.log_path = args['log_path']

		self.history = recorder.LossHistory()

	def make_model(self):	
		inputs = Input(shape=(self.timesteps, self.input_dim))
		encoded = LSTM(self.latent_dim)(inputs)
		
		optioned = RepeatVector(self.option_dim)(encoded)
		optioned = Flatten(optioned)
		
		decoded = RepeatVector(self.timesteps)(optioned)
		decoded = LSTM(self.output_dim*self.option_dim, return_sequences=True)(decoded)
		
		self.encoder = Model(inputs, encoded)
		self.autoencoder = Model(inputs, decoded)
		
		def min_loss(y_true, y_pred):
			ry_pred = K.reshape(y_pred, [self.option_dim, self.output_dim])
			ry_true = K.repeat_elements(y_true, self.option_dim, axis=0)
			return K.min(K.square(ry_pred - ry_true), axis=0)

		self.autoencoder.compile(optimizer='RMSprop', loss=min_loss)

		self.autoencoder.summary()
		self.encoder.summary()

	def run(self, data_iterator): 
		self.make_model()

		if self.trained:
			self.autoencoder.load_weights(self.load_path)
		else:
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
					self.autoencoder.fit(x_train, y_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=(x_test, y_test),
								callbacks=[self.history])

					y_test_decoded = self.autoencoder.predict(x_test[:1])
					image.plot_batch_1D(y_test[:1], y_test_decoded)
					# self.autoencoder.save_weights(self.load_path, overwrite=True)
				iter1, iter2 = tee(iter2)

			data_iterator = iter2
		
		model_vars = [NAME, self.latent_dim, self.timesteps, self.batch_size]
		embedding_plotter.see_embedding(self.encoder, data_iterator, model_vars)
		self.history.record(self.log_path, model_vars)



if __name__ == '__main__':
	data_iterator, config = parser.get_parse(NAME)
	ae = Option_LSTM(config)
	ae.run(data_iterator)
