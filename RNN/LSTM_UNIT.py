import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, LSTM, RepeatVector, Flatten, Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K

from utils import parser, image, embedding_plotter, recorder, option_visualizer


NAME = 'LSTM_UNIT'
USE_GRU = False
if USE_GRU:
	from keras.layers import GRU


class LSTM_UNIT:

	def __init__(self, args):
		self.model = None
		self.translator = None
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
		inputs_1 = Input(shape=(self.timesteps, self.input_dim))
		inputs_2 = Input(shape=(self.timesteps, self.output_dim))

		# inputs_2 = Input(shape=(self.timesteps, self.input_dim))
		# inputs_3 = Input(shape=(self.timesteps, self.output_dim))

		encoder_1 = None
		encoder_2 = None
		if USE_GRU:
			encoder_1 = GRU(self.latent_dim)
			encoder_2 = GRU(self.latent_dim)
		else:
			encoder_1 = LSTM(self.latent_dim)
			encoder_2 = LSTM(self.latent_dim)

		decoder_1 = None
		decoder_2 = None

		if USE_GRU:
			decoder_1 = GRU(self.input_dim, return_sequences=True)
			decoder_2 = GRU(self.output_dim, return_sequences=True)
		else:
			decoder_1 = LSTM(self.input_dim, return_sequences=True)
			decoder_2 = LSTM(self.output_dim, return_sequences=True)

		encoded_1 = encoder_1(inputs_1)
		repeat_1 = RepeatVector(self.timesteps)(encoded_1)
		decoded_1_1 = decoder_1(repeat_1)
		decoded_1_2 = decoder_2(repeat_1)
		
		encoded_2 = encoder_2(inputs_2)
		repeat_2 = RepeatVector(self.timesteps)(encoded_2)
		decoded_2_1 = decoder_1(repeat_2)
		decoded_2_2 = decoder_2(repeat_2)

		# self.encoder_1 = Model(inputs_1, encoded_1)
		# self.encoder_2 = Model(inputs_2, encoded_2)
		self.translator = Model(inputs_1, decoded_1_2)
		self.model = Model([inputs_1, inputs_2], [decoded_1_1, decoded_1_2, decoded_2_1, decoded_2_2])
		self.model.compile(optimizer='RMSprop', loss='mean_squared_error')

		self.model.summary()
		self.translator.summary()

	def load(self):
		self.make_model()
		if self.trained:
			self.model.load_weights(self.load_path)
			return True
		return False

	def run(self, data_iterator): 
		if not self.load():
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
					self.model.fit([x_train, y_train], [x_train, y_train, x_train, y_train],
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							validation_data=([x_test, y_test], [x_test, y_test, x_test, y_test]),
							callbacks=[self.history])

				y_test_decoded = self.translator.predict(x_test[:1])
				image.plot_batch_1D(y_test[:1], y_test_decoded)
				# self.model.save_weights(self.load_path, overwrite=True)
				iter1, iter2 = tee(iter2)
			
			# data_iterator = iter2

		# model_vars = [NAME, self.latent_dim, self.timesteps, self.batch_size]
		# embedding_plotter.see_embedding(self.encoder, data_iterator, model_vars)
		# self.history.record(self.log_path, model_vars)

if __name__ == '__main__':
	data_iterator, config = parser.get_parse(NAME)
	ae = LSTM_UNIT(config)
	ae.run(data_iterator)
