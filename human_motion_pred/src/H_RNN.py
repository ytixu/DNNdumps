import matplotlib
matplotlib.use('Agg')

import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, RepeatVector, Lambda, concatenate
from keras.models import Model
from keras.optimizers import RMSprop

import seq2seq_model__
from utils import parser
from utils import translate__

MODEL_NAME = 'H_GRU'
HAS_LABELS = False
USE_GRU = True

if USE_GRU:
	from keras.layers import GRU as RNN_UNIT
else:
	from keras.layers import LSTM as RNN_UNIT
	MODEL_NAME = 'H_LSTM'

class H_RNN(seq2seq_model__.seq2seq_ae__):

	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.data_dim))
		encoded = RNN_UNIT(self.latent_dim, return_sequences=True)(inputs)

		z = Input(shape=(self.latent_dim,))
		decode_1 = RepeatVector(self.timesteps)
		decode_2 = RNN_UNIT(self.data_dim, return_sequences=True)

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
			for j in range(h+1, self.timesteps):
				y[:,i,j] = y[:,i,h]
		return np.reshape(y, (-1, self.timesteps*len(self.hierarchies), y.shape[-1]))

	def run(self, data_iterator, n=25):
		if not self.load():
			# from keras.utils import plot_model
			# plot_model(self.autoencoder, to_file='model.png')
			loss = 10000
			# iter1, iter2 = tee(data_iterator)
			iterations = 0
			for x in data_iterator:
				print 'iteration', iterations

				x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, x, test_size=self.cv_splits)
				y_train = self.__alter_y(y_train)
				y_test = self.__alter_y(y_test)
				print x_train.shape, x_test.shape, y_train.shape, y_test.shape
				from utils import image
                                xyz = translate__.batch_expmap2xyz(y_train[:5,:5], self)
                                image.plot_poses(xyz)

				history = self.autoencoder.fit(x_train, y_train,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							validation_data=(x_test, y_test))

				print history.history['loss']
				new_loss = np.mean(history.history['loss'])
				if new_loss < loss:
					#self.autoencoder.save_weights(self.save_path, overwrite=True)
					loss = new_loss
					print 'Saved model - ', loss

				y_test_decoded = self.autoencoder.predict(x_test[:1])
				# print y_test_decoded.shape
				y_test_decoded = np.reshape(y_test_decoded, (len(self.hierarchies), self.timesteps, -1))
				self.training_images_plotter(y_test_decoded, x_test[:1])

				if iterations % self.test_every == 0:
					idx = np.random.choice(x_test.shape[0], n, replace=False)
					y_test_encoded = self.encoder.predict(x_test[idx])[:,-1]
					y_test_decoded = self.decoder.predict(y_test_encoded)
					print translate__.euler_diff(x_test[idx], y_test_decoded, self)[0]

				iterations += 1

			# iter1, iter2 = tee(iter2)

			# data_iterator = iter2

if __name__ == '__main__':
	train_set_gen, test_set, config = parser.get_parse(MODEL_NAME, HAS_LABELS)
	ae = H_RNN(config)
	#test_gt, test_pred_gt = test_set
	ae.run(train_set_gen)
