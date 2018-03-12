import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, RepeatVector, Flatten, Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K

from utils import parser, image, embedding_plotter, recorder, option_visualizer


NAME = 'OptionLSTM_2'
USE_GRU = False
if USE_GRU:
	from keras.layers import GRU
else:
	from keras.layers import LSTM


class OptionLSTM_2:

	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		self.translator = None

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
		input_noise = Input(shape=(self.latent_dim,))

		encoded_1 = None
		encoded_2 = None

		if USE_GRU:
			encoded_1 = GRU(self.latent_dim)(inputs_1)
			encoded_2 = GRU(self.latent_dim)(inputs_2, initial_state=encoded_1)
		else:
			encoded_1 = LSTM(self.latent_dim)(inputs_1)
			encoded_2 = LSTM(self.latent_dim)(inputs_2, initial_state=[encoded_1, encoded_1])

		option = concatenate([encoded_1, encoded_2], axis=1)
		option_noise = concatenate([encoded_1, input_noise], axis=1)
		repeater = RepeatVector(self.timesteps)
		decoder = None

		if USE_GRU:
			decoder = GRU(self.output_dim, return_sequences=True)
		else:
			decoder = LSTM(self.output_dim, return_sequences=True)

		decoded = decoder(repeater(option))
		decoded_noise = decoder(repeater(option_noise))

		self.autoencoder = Model([inputs_1, inputs_2], decoded)
		self.encoder = Model(inputs_1, encoded_1)
		self.translator = Model([inputs_1, input_noise], decoded_noise)
		self.autoencoder.compile(optimizer='RMSprop', loss='mean_squared_error')

		self.autoencoder.summary()
		self.encoder.summary()
		self.translator.summary()

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
					self.autoencoder.fit([x_train, y_train], y_train,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							validation_data=([x_test, y_test], y_test),
							callbacks=[self.history])

					y_test_decoded = self.autoencoder.predict([x_test[:5], y_test[:5]])
					y_test_decoded = np.reshape(y_test_decoded, (-1, 1, self.timesteps, self.output_dim))
					image.plot_graph(x_test[:5], y_test[:5], y_test_decoded)

					# image.plot_batch_1D(y_test[:1], y_test_decoded)
					# self.autoencoder.save_weights(self.load_path, overwrite=True)
				iter1, iter2 = tee(iter2)
			
			# data_iterator = iter2

		# model_vars = [NAME, self.latent_dim, self.timesteps, self.batch_size]
		# embedding_plotter.see_embedding(self.encoder, data_iterator, model_vars)
		# self.history.record(self.log_path, model_vars)

if __name__ == '__main__':
	data_iterator, config = parser.get_parse(NAME)
	ae = OptionLSTM_2(config)
	ae.run(data_iterator)
