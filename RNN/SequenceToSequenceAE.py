from sklearn import cross_validation
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

from utils import parser, image

class LSTM_AE:

	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		# self.decoder = None

		self.timesteps = 0
		self.input_dim = 0
		self.output_dim = 0

		self.epochs = args['epochs'] if 'epochs' in args else 10
		self.periods = args['periods'] if 'periods' in args else 5
		self.batch_size = args['batch_size'] if 'batch_size' in args else 16
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 5
		self.input_dim = args['input_dim']
		self.output_dim = args['output_dim']
		self.latent_dim = (self.input_dim + self.output_dim) / 2


	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.input_dim))
		encoded = LSTM(self.latent_dim)(inputs)

		decoded = RepeatVector(self.timesteps)(encoded)
		decoded = LSTM(self.output_dim, return_sequences=True)(decoded)

		self.autoencoder = Model(inputs, decoded)
		self.encoder = Model(inputs, encoded)
		# self.decoder = Model(encoded, decoded)
		self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

		self.autoencoder.summary()
		self.encoder.summary()
		# self.decoder.summary()


	def run(self, data_iterator): 
		self.make_model()

		for x, y in data_iterator:
			for i in range(self.periods):
				x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
				self.autoencoder.fit(x_train, y_train,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							validation_data=(x_test, y_test))

				y_test_decoded = self.autoencoder.predict(x_test[:1])
				image.plot_batch(y_test[:1], y_test_decoded)


if __name__ == '__main__':
	data_iterator, config = parser.get_parse('LSTM_AE')
	ae = LSTM_AE(config)
	ae.run(data_iterator)
