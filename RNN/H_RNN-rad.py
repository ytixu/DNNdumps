import matplotlib
matplotlib.use('Agg')

import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, RepeatVector, Lambda, concatenate
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop

from utils import parser, image, embedding_plotter, metrics, metric_baselines, fk_animate, association_evaluation, evaluate

NAME = 'H_LSTM_R'
USE_GRU = True
L_RATE = 0.001

if USE_GRU:
	from keras.layers import GRU as RNN_UNIT
	NAME = 'H_GRU_R'
else:
	from keras.layers import LSTM as RNN_UNIT

def wrap_angle(rad):
	return ( rad + np.pi) % (2 * np.pi ) - np.pi

def normalize_angle(rad):
	return rad/np.pi

def unormalize_angle(rad):
	return rad*np.pi

class H_RNN_R:
	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		self.decoder = None

		self.epochs = args['epochs']
		self.batch_size = args['batch_size']
		self.periods = args['periods'] if 'periods' in args else 10
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 10
		self.hierarchies = args['hierarchies'] if 'hierarchies' in args else range(self.timesteps)
		# self.hierarchies = args['hierarchies'] if 'hierarchies' in args else range(self.timesteps)
		self.input_dim = args['input_dim']
		self.output_dim = args['output_dim']
		self.latent_dim = args['latent_dim'] if 'latent_dim' in args else (args['input_dim']+args['output_dim'])/2
		self.trained = args['mode'] == 'sample' if 'mode' in args else False
		self.load_path = args['load_path']
		self.save_path = args['save_path']
		self.log_path = args['log_path']

		self.used_euler_idx = [6,7,8,9,12,13,14,15,21,22,23,24,27,28,29,30,36,37,38,39,40,41,42,43,44,45,46,47,51,52,53,54,55,56,57,60,61,62,75,76,77,78,79,80,81,84,85,86]
		self.used_xyz_idx =   [3,4,5,6, 7, 8, 9,10,11,18,19,20,21,22,23,24,25,26,36,37,38,39,40,41,42,43,44,45,46,47,51,52,53,54,55,56,57,58,59,75,76,77,78,79,80,81,82,83]
		self.euler_start = len(self.used_xyz_idx)

		self.MODEL_CODE = metrics.H_RNN_R

		# self.history = recorder.LossHistory()

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
		opt = RMSprop(lr=L_RATE)
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


	def __merge_n_reparameterize(self, x, y):
		y = normalize_angle(wrap_angle(y[:,:,self.used_euler_idx]))
		x = x[:,:,self.used_xyz_idx]
		return np.concatenate([x,y], -1)

	def euler_error(self, yTrue, yPred):
		return np.mean(np.sqrt(np.sum(np.square(yTrue - yPred), -1)), 0)

	def run(self, data_iterator, valid_data):
		# model_vars = [NAME, self.latent_dim, self.timesteps, self.batch_size]
		if self.load():
			# from keras.utils import plot_model
			# plot_model(self.autoencoder, to_file='model.png')
			loss = 10000
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					x = self.__merge_n_reparameterize(x,y)
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, x, test_size=self.cv_splits)
					y_train = self.__alter_y(y_train)
					y_test = self.__alter_y(y_test)
					history = self.autoencoder.fit(x_train, y_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=(x_test, y_test))
								# callbacks=[tbCallBack])

					print history.history['loss']
					new_loss = np.mean(history.history['loss'])
					if new_loss < loss:
						# self.autoencoder.save_weights(self.save_path, overwrite=True)
						loss = new_loss
						print 'Saved model - ', loss

					rand_idx = np.random.choice(x.shape[0], 25, replace=False)
					y_test_pred = self.encoder.predict(x[rand_idx])[:,-1]
					y_test_pred = self.decoder.predict(y_test_pred)[:,:,self.euler_start:]

					y_test_pred = self.unormalize_angle(y_test_pred)
					y_gt = wrap_angle(y[rand_idx,:,self.used_euler_idx])

					mae = np.mean(np.abs(y_gt-y_test_pred))
					mse = self.euler_error(y_gt, y_test_pred)

					print 'MAE', mae
					print 'MSE', mse

				iter1, iter2 = tee(iter2)

			data_iterator = iter2

if __name__ == '__main__':
	data_iterator, valid_data, config = parser.get_parse(NAME)
	ae = H_RNN_R(config)
	ae.run(data_iterator, valid_data)
