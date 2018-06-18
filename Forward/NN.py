import numpy as np
import time
from keras.layers import Input, Dense
from keras.models import Model
from sklearn import cross_validation

NAME = 'Forward_NN'

class Forward_NN:

	def __init__(self, args):
		self.batch_size = args['batch_size']
		self.epochs = args['epochs']
		self.output_dim = args['output_dim']
		self.input_dim = args['input_dim']
		self.interim_dim = (args['output_dim'] + args['input_dim'])/2
		self.layers = args['layers']
		self.epsilon_std = 1.0
		self.periods = args['periods'] if 'periods' in args else 10
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2
		self.trained = args['mode'] == 'sample'

		self.load_path = args['load_path']
		self.save_path = args['save_path']

		self.model = None

	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.input_dim))
		d1 = Dense(self.interim_dim, activation='relu')(inputs)
		ouputs = Dense(self.output_dim, activation='tanh')(d1)

		self.model = Model(inputs, outputs)
		self.model.compile(optimizer='RMSprop', loss='mean_squared_error')

	def load(self):
		self.make_model()
		if self.trained:
			self.model.load_weights(self.load_path)
			return True
		return False

	def run(self, x, y):
		if not self.load():
			loss_error = 10000
			for i in range(self.periods):
				x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
				history = self.autoencoder.fit(x_train, y_train,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							validation_data=(x_test, y_test))
							# callbacks=[self.history])

				new_loss = np.mean(history.history['loss'])
				if new_loss < loss:
					print 'Saved model - ', loss
					loss = new_loss
					self.model.save_weights(self.save_path, overwrite=True)



if __name__ == '__main__':
	x, y, config = parser.get_parse(NAME, labels=True)
	nn = Forward_NN(config)
	nn.run(data_iterator)
