import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from keras.models import Model, Sequential

from utils import parser, image

class H_LSTM:

	def __init__(self, args):
		self.model = None

		self.epochs = args['epochs'] if 'epochs' in args else 10
		self.periods = args['periods'] if 'periods' in args else 5
		self.batch_size = args['batch_size'] if 'batch_size' in args else 16
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 5
		self.level_n = args['level_n'] if 'level_n' in args else 5
		self.input_dim = args['input_dim']
		self.output_dim = args['output_dim']
		self.trained = args['mode'] == 'sample' if 'mode' in args else False


	def make_model(self):
		inputs = Input(shape=(self.level_n, self.timesteps, self.input_dim))
		encoded_rows = TimeDistributed(LSTM(self.input_dim))(inputs)
		encoded_columns = LSTM(self.input_dim)(encoded_rows)

		decoded_columns = RepeatVector(self.timesteps)(encoded_columns)
		decoded_columns = LSTM(self.output_dim)(decoded_columns)
		decoded__rows = RepeatVector(self.timesteps)(decoded_columns)
		decoded_rows = LSTM(self.output_dim, return_sequences=True)(decoded__rows)
		
		self.model = Model(inputs, decoded_rows)
		self.model.compile(optimizer='RMSprop', loss='mean_squared_error')

		self.model.summary()

	
	def run(self, data_iterator): 
		self.make_model()

		if self.trained:
			self.model.load_weights(self.load_path)
			# blah

		else:
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					norm_y = y/np.pi/2
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, norm_y, test_size=self.cv_splits)
					self.model.fit(x_train, y_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=(x_test, y_test))

					y_test_decoded = self.model.predict(x_test[:1])
					image.plot_batch_1D(y_test[:1], y_test_decoded)
					# self.model.save_weights(self.load_path, overwrite=True)
				
				iter1, iter2 = tee(iter2)

if __name__ == '__main__':
	data_iterator, config = parser.get_parse('LSTM_AE')
	ae = H_LSTM(config)
	ae.run(data_iterator)
