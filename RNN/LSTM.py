import numpy as np
from sklearn import cross_validation
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

from utils import parser, image

class LSTM_AE:

	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		# self.decoder = None

		self.epochs = args['epochs'] if 'epochs' in args else 10
		self.periods = args['periods'] if 'periods' in args else 5
		self.batch_size = args['batch_size'] if 'batch_size' in args else 16
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 5
		self.input_dim = args['input_dim']
		self.output_dim = args['output_dim']
		self.latent_dim = (self.timesteps + self.input_dim + self.output_dim)/3
		self.trained = args['mode'] == 'sample' if 'mode' in args else False


	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.input_dim))
		encoded = LSTM(self.latent_dim)(inputs)

		decoded = RepeatVector(self.timesteps)(encoded)
		decoded = LSTM(self.output_dim, return_sequences=True)(decoded)
		
		self.encoder = Model(inputs, encoded)
		self.autoencoder = Model(inputs, decoded)
		self.autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

		self.autoencoder.summary()
		self.encoder.summary()

	def run(self, data_iterator): 
		self.make_model()

		if self.trained:
			self.autoencoder.load_weights(self.load_path)
			# blah

		else:
			for x, y in data_iterator:
				norm_y = y/np.pi/2
				for i in range(self.periods):
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, norm_y, test_size=self.cv_splits)
					self.autoencoder.fit(x_train, y_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=(x_test, y_test))

					y_test_decoded = self.autoencoder.predict(x_test[:1])
					image.plot_batch_1D(y_test[:1], y_test_decoded)
					# self.autoencoder.save_weights(self.load_path, overwrite=True)

if __name__ == '__main__':
	data_iterator, config = parser.get_parse('LSTM_AE')
	ae = LSTM_AE(config)
	ae.run(data_iterator)
