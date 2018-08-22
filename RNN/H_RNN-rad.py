import matplotlib
matplotlib.use('Agg')

import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, RepeatVector, Lambda, concatenate
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
import csv
from tqdm import tqdm
import json

from utils import parser, image, embedding_plotter, metrics, metric_baselines, fk_animate, association_evaluation, evaluate

NAME = 'H_LSTM_R'
USE_GRU = True
L_RATE = 0.0005

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
		self.hierarchies = args['hierarchies'] if 'hierarchies' in args else [14,24]
		# self.hierarchies = args['hierarchies'] if 'hierarchies' in args else range(self.timesteps)
		self.latent_dim = args['latent_dim'] if 'latent_dim' in args else (args['input_dim']+args['output_dim'])/2
		self.trained = args['mode'] == 'sample' if 'mode' in args else False
		self.load_path = args['load_path']
		self.save_path = args['save_path']
		self.log_path = args['log_path']

		self.used_euler_idx = [6,7,8,9,12,13,14,15,21,22,23,24,27,28,29,30,36,37,38,39,40,41,42,43,44,45,46,47,51,52,53,54,55,56,57,60,61,62,75,76,77,78,79,80,81,84,85,86]
		self.used_xyz_idx =   [3,4,5,6, 7, 8, 9,10,11,18,19,20,21,22,23,24,25,26,36,37,38,39,40,41,42,43,44,45,46,47,51,52,53,54,55,56,57,58,59,75,76,77,78,79,80,81,82,83]
		self.euler_start = len(self.used_xyz_idx)

		self.input_dim = len(self.used_euler_idx) + len(self.used_xyz_idx)
                self.output_dim = self.input_dim

		self.MODEL_CODE = metrics.H_LSTM

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


	def __merge_n_reparameterize(self, x, y, return_orig=False):
		y = normalize_angle(wrap_angle(y[:,:,self.used_euler_idx]))
		x = x[:,:,self.used_xyz_idx]
		xy = np.concatenate([x,y], -1)
		if return_orig:
			return x, y, xy
		return xy

	def euler_error(self, yTrue, yPred):
		return np.mean(np.sqrt(np.sum(np.square(yTrue - yPred), -1)), 0)

	def run(self, data_iterator, valid_data):
		# model_vars = [NAME, self.latent_dim, self.timesteps, self.batch_size]
		if not self.load():
			# from keras.utils import plot_model
			# plot_model(self.autoencoder, to_file='model.png')
			loss = 10000
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					image.plot_fk_from_euler(x[:3], title='test')
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
						self.autoencoder.save_weights(self.save_path, overwrite=True)
						loss = new_loss
						print 'Saved model - ', loss

					rand_idx = np.random.choice(x.shape[0], 25, replace=False)
					y_test_pred = self.encoder.predict(x[rand_idx])[:,-1]
					y_test_pred = self.decoder.predict(y_test_pred)[:,:,self.euler_start:]

					y_test_pred = unormalize_angle(y_test_pred)
					y_gt = wrap_angle(y[rand_idx][:,:,self.used_euler_idx])

					mae = np.mean(np.abs(y_gt-y_test_pred))
					mse = self.euler_error(y_gt, y_test_pred)

					print 'MAE', mae
					print 'MSE', mse

					with open('../new_out/%s_t%d_l%d_log.csv'%(NAME, self.timesteps, self.latent_dim), 'a+') as f:
						spamwriter = csv.writer(f)
						spamwriter.writerow([new_loss, mae, mse, L_RATE])


				iter1, iter2 = tee(iter2)

			data_iterator = iter2
		else:
			# load embedding
			embedding = []
			for x,y in data_iterator:
				x = self.__merge_n_reparameterize(x,y)
				e = self.encoder.predict(x)
				if len(embedding) == 0:
					embedding = e[:, self.hierarchies]
				else:
					embedding = np.concatenate((embedding, e[:,self.hierarchies]), axis=0)
				#break
			embedding = np.array(embedding)
			print 'emb', embedding.shape
			mean_diff, diff = metrics.get_embedding_diffs(embedding[:,1], embedding[:,0])

			_N = 100
			methods = ['closest_partial', 'closest', 'add']
			cut = self.hierarchies[0]
			pred_n = self.hierarchies[1]-cut
			error = {m: {'euler': np.zeros(pred_n),
					'pose': np.zeros(pred_n)}  for m in methods}

			x, y = valid_data
			rand_idx = np.random.choice(x.shape[0], _N, replace=False)
			x, y, valid_data = self.__merge_n_reparameterize(x[rand_idx],y[rand_idx], True)
			y = unormalize_angle(y)
			enc = self.encoder.predict(valid_data)
			partial_enc = enc[:,cut]

			# autoencoding error for partial seq
			dec = self.decoder.predict(partial_enc)[:,:cut+1]
			dec_euler = unormalize_angle(dec[:,:,self.euler_start:])
			print self.euler_error(y[:,:cut+1], dec_euler)
			#image.plot_poses_euler(x[:2,:cut+1], dec[:2,:,:self.euler_start], title='autoencoding', image_dir='../new_out/')

			for method in methods:
				new_enc = np.zeros(partial_enc.shape)
				for i in tqdm(range(_N)):
					if method == 'closest_partial':
						idx = metrics.closest_partial_index(embedding[:,0], partial_enc[i])
						new_enc[i] = embedding[idx,1]
					elif method == 'closest':
						new_enc[i] = metrics.closest(embedding[:,1], partial_enc[i])
					elif method == 'add':
						new_enc[i] = partial_enc[i]+mean_diff

				model_pred = self.decoder.predict(new_enc)[:,cut+1:]
				model_pred_euler = unormalize_angle(model_pred[:,:,self.euler_start:])
				error[method]['euler'] = self.euler_error(y[:,cut+1:], model_pred_euler)
				print method
				print error[method]['euler']

				image.plot_poses_euler(x[:2,cut+1:], model_pred[:2,:,:self.euler_start], title=method, image_dir='../new_out/')

				for i in range(_N):
					pose_err = metrics.pose_seq_error(x[i,cut+1:], model_pred[i,:,:self.euler_start], cumulative=True)
					error[method]['pose'] = error[method]['pose'] + np.array(pose_err)
				error[method]['pose'] = error[method]['pose']/_N

				print error[method]['pose']
				error[method]['euler'] = error[method]['euler'].tolist()
				error[method]['pose'] = error[method]['pose'].tolist()


			with open('../new_out/%s_t%d_l%d_validation.json'%(NAME, self.timesteps, self.latent_dim), 'wb') as result_file:
				json.dump(error, result_file)



if __name__ == '__main__':
	data_iterator, valid_data, config = parser.get_parse(NAME)
	ae = H_RNN_R(config)
	ae.run(data_iterator, valid_data)
