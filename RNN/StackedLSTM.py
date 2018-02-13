import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model, Sequential

from utils import parser, image

class Stacked_LSTM:

	def __init__(self, args):
		self.model = None

		self.epochs = args['epochs'] if 'epochs' in args else 10
		self.periods = args['periods'] if 'periods' in args else 5
		self.batch_size = args['batch_size'] if 'batch_size' in args else 16
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 5
		self.input_dim = args['input_dim']
		self.output_dim = args['output_dim']
		# self.latent_dim = (self.timesteps + self.input_dim + self.output_dim)/3
		self.trained = args['mode'] == 'sample' if 'mode' in args else False


	def make_model(self):
		self.model = Sequential()
		self.model.add(LSTM(self.input_dim, return_sequences=True, input_shape=(self.timesteps, self.input_dim))) 
		# self.model.add(LSTM(self.latent_dim, return_sequences=True)) 
		self.model.add(LSTM(self.input_dim)) 
		self.model.add(RepeatVector(self.timesteps))
		self.model.add(LSTM(self.output_dim, return_sequences=True))
		self.model.add(LSTM(self.output_dim, return_sequences=True))
		self.model.compile(optimizer='adam', loss='mean_squared_error')

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
	ae = Stacked_LSTM(config)
	ae.run(data_iterator)
