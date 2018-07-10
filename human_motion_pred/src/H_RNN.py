import matplotlib
matplotlib.use('Agg')

import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, RepeatVector, Lambda, concatenate
from keras.models import Model
from keras.optimizers import RMSprop

import seq2seq_model
import parser

NAME = 'H_GRU'
USE_GRU = True
RNN_UNIT = GRU

if USE_GRU:
	from keras.layers import GRU
else:
	from keras.layers import LSTM
	NAME = 'H_LSTM'
	RNN_UNIT = LSTM

class H_RNN(seq2seq_model.seq2seq_ae__):

	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.input_dim))
		encoded = RNN_UNIT(self.latent_dim, return_sequences=True)(inputs)

		z = Input(shape=(self.latent_dim,))
		decode_1 = RepeatVector(self.timesteps)
		decode_2 = RNN_UNIT(self.output_dim, return_sequences=True)

		decoded = [None]*len(self.hierarchies)
		if len(self.hierarchies) == 1:
			e = Lambda(lambda x: x[:,self.hierarchies[0]], output_shape=(self.latent_dim,))(encoded)
			decoded = decode_1(e)
                        decoded = decode_2(decoded)
		else:
			for i, h in enumerate(self.hierarchies):
				e = Lambda(lambda x: x[:,h], output_shape=(self.latent_dim,))(encoded)
				decoded[i] = decode_1(e)
				decoded[i] = decode_2(decoded[i])
			decoded = concatenate(decoded, axis=1)

		decoded_ = decode_1(z)
		decoded_ = decode_2(decoded_)

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.autoencoder = Model(inputs, decoded)
		opt = RMSprop(lr=self.lr)
		self.autoencoder.compile(optimizer=opt, loss='mean_squared_error')

		self.autoencoder.summary()
		self.encoder.summary()
		self.decoder.summary()

	def load(self):
		self.make_model()
		if self.trained:
			self.autoencoder.load_weights(self.load_path)
			print 'LOADED------'
			return True
		return False

	def __alter_y(self, y):
		if len(self.hierarchies) == 1:
			return y
		y = np.repeat(y, len(self.hierarchies), axis=0)
		y = np.reshape(y, (-1, len(self.hierarchies), self.timesteps, y.shape[-1]))
		for i, h in enumerate(self.hierarchies):
			y[:,i,h+1:,:] = 0.0
		return np.reshape(y, (-1, self.timesteps*len(self.hierarchies), y.shape[-1]))

	def run(self, data_iterator, valid_data):
		if self.load():
			# from keras.utils import plot_model
			# plot_model(self.autoencoder, to_file='model.png')
			loss = 10000
			# iter1, iter2 = tee(data_iterator)
			for x, y in data_iterator:
				x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
				y_train = self.__alter_y(y_train)
				y_test_orig = np.copy(y_test[:1])
				y_test = self.__alter_y(y_test)
				history = self.autoencoder.fit(x_train, y_train,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							validation_data=(x_test, y_test))

				print history.history['loss']
				new_loss = np.mean(history.history['loss'])
				if new_loss < loss:
					self.autoencoder.save_weights(self.save_path, overwrite=True)
					loss = new_loss
					print 'Saved model - ', loss

				y_test_decoded = self.autoencoder.predict(x_test[:1])
				# print y_test_decoded.shape
				y_test_decoded = np.reshape(y_test_decoded, (len(self.hierarchies), self.timesteps, -1))
				self.training_images_plotter(y_test_decoded, x_test[:1])

			# iter1, iter2 = tee(iter2)

			# data_iterator = iter2

if __name__ == '__main__':
	train_set_gen, test_set, config = parser.get_parse(MODEL_NAME, HAS_LABELS)
	ae = H_RNN(config)
	test_gt, test_pred_gt = test_set
	ae.run(train_set_gen)