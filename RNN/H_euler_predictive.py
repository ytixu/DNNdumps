import matplotlib
matplotlib.use('Agg')

import numpy as np
from itertools import tee
from sklearn import cross_validation
import keras.layers as K_layer
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import optimizers # import Nadam
import csv
from tqdm import tqdm
import json
import keras.backend as K
import tensorflow as tf

from utils import parser, image, embedding_plotter, metrics, metric_baselines, fk_animate, association_evaluation, evaluate
from Forward import NN

NAME = 'H_euler_Pred_LSTM_R'
USE_GRU = True
L_RATE = 0.001

if USE_GRU:
	from keras.layers import GRU as RNN_UNIT
	NAME = 'H_euler_Pred_GRU_R'
else:
	from keras.layers import LSTM as RNN_UNIT

def wrap_angle(rad, center=0):
	return ( rad - center + np.pi) % (2 * np.pi ) - np.pi


class H_euler_Pred_RNN_R:
	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		self.decoder = None

		self.epochs = args['epochs']
		self.batch_size = args['batch_size']
		self.periods = args['periods'] if 'periods' in args else 10
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2
		self.trained = args['mode'] == 'sample' if 'mode' in args else False
		self.timesteps_out = 10
		self.timesteps = (args['timesteps'] if 'timesteps' in args else 20) - self.timesteps_out
		self.partial_ts = 5
		self.partial_n = self.timesteps/self.partial_ts
		self.hierarchies = range(self.partial_ts-1,self.timesteps, self.partial_ts)
		#[14,24] if self.trained else range(self.partial_ts-1,self.timesteps, self.partial_ts)
		self.predict_hierarchies = [2,4]
		# self.hierarchies = args['hierarchies'] if 'hierarchies' in args else range(self.timesteps)
		self.latent_dim = args['latent_dim'] if 'latent_dim' in args else (args['input_dim']+args['output_dim'])/2
		self.load_path = args['load_path']
		self.save_path = args['save_path']
		self.log_path = args['log_path']

		self.loss_func = args['loss_func']
		self.opt = eval(args['optimizer'])
		self.loss_opt_str = self.loss_func + '_' + args['optimizer'].replace('(', '-').replace(')', '').replace(',','-')

		self.used_euler_idx = [6,7,8,9,12,13,14,15,21,22,23,24,27,28,29,30,36,37,38,39,40,41,42,43,44,45,46,47,51,52,53,54,55,56,57,60,61,62,75,76,77,78,79,80,81,84,85,86]
		self.ignored_idx = list(set(range(99)) - set(self.used_euler_idx))

		self.MODEL_CODE = metrics.H_LSTM

		self.get_ignored_dims()

	def get_ignored_dims(self):
		import json
		with open('../data/h3.6/full/stats_euler.json') as data_file:
			data = json.load(data_file)
			#self.used_euler_idx = data['dim_to_use']
			self.data_mean_all = np.array(data['data_mean'])
			self.data_mean = self.data_mean_all[self.used_euler_idx]

		self.input_dim = len(self.used_euler_idx)
		self.output_dim = self.input_dim
		print 'data_dim', self.input_dim

	def normalize_angle(self, rad):
		return wrap_angle(rad, self.data_mean)/np.pi

	def unormalize_angle(self, rad):
		return wrap_angle(rad*np.pi, -self.data_mean)

	def make_model(self):
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))

		reshaped = K_layer.Reshape((self.partial_n, self.partial_ts, self.input_dim))(inputs)
		encode_reshape = K_layer.Reshape((self.partial_n, self.latent_dim/2))
		encode_1 = RNN_UNIT(self.latent_dim/2)
		encode_2 = RNN_UNIT(self.latent_dim)

		def encode_partials(seq):
			encoded = [None]*self.partial_n
			for i in range(self.partial_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.partial_ts, self.input_dim))(seq)
				encoded[i] = encode_1(rs)
			return encode_reshape(K_layer.concatenate(encoded, axis=1))

		encoded = encode_partials(reshaped)
		print K.int_shape(encoded), K.int_shape(reshaped)
		encoded = encode_2(encoded)

		z = K_layer.Input(shape=(self.latent_dim,))
		decoder_activation = 'tanh'
		decode_euler_1 = K_layer.Dense(self.latent_dim/2, activation=decoder_activation)
		decode_euler_2 = K_layer.Dense(self.output_dim, activation=decoder_activation)

		decode_repete = K_layer.RepeatVector(self.timesteps_out)
		decode_residual_1 = RNN_UNIT(self.latent_dim/2, return_sequences=True, activation=decoder_activation)
		decode_residual_2 = RNN_UNIT(self.output_dim, return_sequences=True, activation=decoder_activation)

		def decode_angle(e):
			angle = decode_euler_2(decode_euler_1(e))
			residual = decode_repete(e)
			residual = decode_residual_2(decode_residual_1(residual))
			angle = K_layer.Activation(decoder_activation)(K_layer.add([decode_repete(angle), residual]))
			return angle


		# angles = [None]*self.partial_n
		# for i in range(self.partial_n):
		# 	e = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.latent_dim,))(encoded)
		# 	angles[i] = decode_angle(e)
		# decoded =  K_layer.concatenate(angles, axis=1)
		decoded = decode_angle(encoded)
		decoded_ = decode_angle(z)

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.autoencoder = Model(inputs, decoded)

		# def mse(yTrue, yPred):
		# 	return K.mean(K.sqrt(K.sum(K.square(yTrue - yPred), -1)), 0)

		self.autoencoder.compile(optimizer=self.opt, loss=self.loss_func)

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

	def __alter_data(self, x):
		return x[:,:self.timesteps], x[:,self.timesteps:]

	def __alter_parameterization(self, y):
		y = y[:,:,self.used_euler_idx]
		norm_y = self.normalize_angle(y)
		return y, norm_y

	def __reparameterize(self, y):
		yt = np.zeros((y.shape[0], y.shape[1], 99))
		#0 - 6 are 0
		yt[:,:,6:] = wrap_angle(self.data_mean_all[6:])
		yt[:,:,self.used_euler_idx] = y
		return yt

	def euler_error(self, yTrue, yPred):
		#yTrue[:,:,:6] = 0
		# from Matinez
		#yPred = self.__reparameterize(yPred)
		return np.mean(np.sqrt(np.sum(np.square(yTrue - yPred), -1)), 0)
		# from Tang
		yPred = np.reshape(self.__reparameterize(yPred), (-1, yPred.shape[1], 33, 3))
		yTrue = np.reshape(yTrue, (-1, yTrue.shape[1], 33, 3))
		print np.sum(np.sqrt(np.sum(np.square(yTrue - yPred), -1)), -1).shape
		return np.mean(np.sum(np.sqrt(np.sum(np.square(yTrue - yPred), -1)), -1), 0)

	def load_validation_data(self, load_path):
		x = [None]*15
		y = [None]*15
		i = 0
		for basename in metric_baselines.iter_actions():
			x[i] = np.load(load_path + 'euler/' + basename + '_cond.npy')[:,-self.timesteps:]
			y[i] = np.load(load_path + 'euler/' + basename + '_gt.npy')[:,:self.timesteps_out]
			i += 1
		x = np.concatenate(x, axis=0)
		y = np.concatenate(y, axis=0)
		# image.plot_fk_from_euler(y[:2,10:20], title='test')
		# image.plot_fk_from_euler(y[2:4,10:20], title='test')

		_,x = self.__alter_parameterization(x)
		y = wrap_angle(y)
		return y, x


	def validate(self, test_data_x, test_data_y):
		y_test_pred = self.autoencoder.predict(test_data_x)
		y_test_pred = self.unormalize_angle(y_test_pred)
		return self.euler_error(test_data_y, y_test_pred)

	def run(self, data_iterator, valid_data):
		load_path = '../human_motion_pred/baselines/'
		if not self.load():
			test_data_y, test_data_x = self.load_validation_data(load_path)
			print self.loss_opt_str
			# from keras.utils import plot_model
			# plot_model(self.autoencoder, to_file='model.png')
			loss = 10000
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, _ in iter1:
					#image.plot_fk_from_euler(x[:3], title='test')
					x_orig, x = self.__alter_parameterization(x)
					x, y = self.__alter_data(x)
					_, y_orig = self.__alter_data(x_orig)
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)

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

					rand_idx = np.random.choice(x.shape[0], 100, replace=False)
					y_gt = wrap_angle(y_orig[rand_idx])

					mse = self.validate(x[rand_idx], y_gt)
					mse_test = self.validate(test_data_x, test_data_y)
					mse_pred = self.validate(test_data_x, test_data_y)

					print 'MSE', np.mean(mse)
					print 'MSE TEST', np.mean(mse_test)
					print 'MSE PRED', mse_pred[-10:]

					with open('../new_out/%s_t%d_l%d_%s_log.csv'%(NAME, self.timesteps, self.latent_dim, self.loss_opt_str), 'a+') as f:
						spamwriter = csv.writer(f)
						spamwriter.writerow([new_loss, mse, mse_test, mse_pred, self.loss_opt_str])

				iter1, iter2 = tee(iter2)

			data_iterator = iter2
		else:
			_N = 8
			error = {act: 0 for act in metric_baselines.iter_actions()}

			# a_n = 0
			for basename in metric_baselines.iter_actions():
				print basename, '================='

				x = np.load(load_path + 'euler/' + basename + '_cond.npy')[:,-self.timesteps:]
				y = np.load(load_path + 'euler/' + basename + '_gt.npy')[:,:self.timesteps_out]
				_,x = self.__alter_parameterization(x)
				y = wrap_angle(y)

				mse_pred = self.validate(test_data_x, test_data_y)
				error[basename] = self.validate(test_data_x, test_data_y).tolist()
				print error[basename]

				#image.plot_poses_euler(gtp_x[:2], model_pred[:2,:,:self.euler_start], title=method, image_dir='../new_out/')

				# error[method]['z'] = np.mean([np.linalg.norm(new_enc[i] - enc[i,-1]) for i in range(_N)])
				# print error[method]['z']

				# for i in range(_N):
				# 	pose_err = metrics.pose_seq_error(gtp_x[i], model_pred[i,:,:self.euler_start], cumulative=True)
				# 	error[method]['pose'] = error[method]['pose'] + np.array(pose_err)
				# error[method]['pose'] = error[method]['pose']/_N
				# print error[method]['pose']
				# error[method]['pose'] = error[method]['pose'].tolist()


			with open('../new_out/%s_t%d_l%d_opt-%s_validation-testset-mseMartinez.json'%(NAME, self.timesteps, self.latent_dim, self.loss_opt_str), 'wb') as result_file:
				json.dump(error, result_file)



if __name__ == '__main__':
	data_iterator, valid_data, config = parser.get_parse(NAME)
	ae = H_euler_Pred_RNN_R(config)
	ae.run(data_iterator, valid_data)
