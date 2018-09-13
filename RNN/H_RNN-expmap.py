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

NAME = 'H_expmap_LSTM_R_h2'
USE_GRU = True
L_RATE = 0.001

if USE_GRU:
	from keras.layers import GRU as RNN_UNIT
	NAME = 'H_expmap_GRU_R_h2'
else:
	from keras.layers import LSTM as RNN_UNIT

def expmap2rotmat(r):
	theta = np.linalg.norm( r )
	r0  = np.divide( r, theta + np.finfo(np.float32).eps )
	r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
	r0x = r0x - r0x.T
	R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);
	return R

def rotmat2euler( R ):
	if R[0,2] == 1 or R[0,2] == -1:
		# special case
		E3   = 0 # set arbitrarily
		dlta = np.arctan2( R[0,1], R[0,2] );
		if R[0,2] == -1:
			E2 = np.pi/2;
			E1 = E3 + dlta;
		else:
			E2 = -np.pi/2;
			E1 = -E3 + dlta;
	else:
		E2 = -np.arcsin( R[0,2] )
		E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
		E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

	eul = np.array([E1, E2, E3]);
	return eul

EULER_IDEX = [6,7,8,9,12,13,14,15,21,22,23,24,27,28,29,30,36,37,38,39,40,41,42,43,44,45,46,47,51,52,53,54,55,56,57,60,61,62,75,76,77,78,79,80,81,84,85,86]


class H_expmap_RNN_R:
	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		self.decoder = None

		self.epochs = args['epochs']
		self.batch_size = args['batch_size']
		self.periods = args['periods'] if 'periods' in args else 10
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2
		self.trained = args['mode'] == 'sample' if 'mode' in args else False
		self.timesteps = args['timesteps'] if 'timesteps' in args else 10
		self.partial_ts = 10
		self.partial_n = self.timesteps/self.partial_ts
		self.hierarchies = range(self.partial_ts-1,self.timesteps, self.partial_ts)
		self.hierarchies_index = range(self.partial_n)
		#[14,24] if self.trained else range(self.partial_ts-1,self.timesteps, self.partial_ts)
		self.predict_hierarchies = [0,1]
		# self.hierarchies = args['hierarchies'] if 'hierarchies' in args else range(self.timesteps)
		self.latent_dim = args['latent_dim'] if 'latent_dim' in args else (args['input_dim']+args['output_dim'])/2
		self.load_path = args['load_path']
		self.save_path = args['save_path']
		self.log_path = args['log_path']

		self.loss_func = args['loss_func']
		self.opt = eval(args['optimizer'])
		self.loss_opt_str = self.loss_func + '_' + args['optimizer'].replace('(', '-').replace(')', '').replace(',','-')

		self.MODEL_CODE = metrics.H_LSTM

		self.get_ignored_dims()

	def get_ignored_dims(self):
		import json
		with open('../data/h3.6/raw/h3.6m/params.json') as data_file:
			data = json.load(data_file)
			self.used_expmap_idx = np.array(data['dim_to_use'])
			self.data_mean = np.array(data['data_mean'])

		with open('../data/h3.6/full/stats_expmap.json') as data_file:
			data = json.load(data_file)
			self.data_max = np.ceil(data['data_max'])
			self.data_min = np.floor(data['data_min'])

		self.input_dim = len(self.used_expmap_idx)
		self.output_dim = self.input_dim
		print 'data_dim', self.input_dim

	def normalize_angle(self, expmap):
		return 2*(expmap-self.data_min)/(self.data_max-self.data_min)-1

	def unormalize_angle(self, expmap):
		return (expmap+1)/2*(self.data_max-self.data_min)+self.data_min

	def recover(self, expmap):
		rec_expmap = np.zeros((expmap.shape[0], expmap.shape[1], self.data_mean.shape[0]))
		rec_expmap[:,:] = self.data_mean
		rec_expmap[:,:,self.used_expmap_idx] = expmap
		return rec_expmap

	def make_model(self):
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		#sin_layer = K_layer.Lambda(lambda x: K.sin(x), output_shape=(self.timesteps, self.input_dim))
		#cos_layer = K_layer.Lambda(lambda x: K.cos(x), output_shape=(self.timesteps, self.input_dim))
		#decomposed = K_layer.concatenate([sin_layer(inputs), cos_layer(inputs)], axis=1)
		reshaped = K_layer.Reshape((self.partial_n, self.partial_ts, self.input_dim))(inputs)
		encode_reshape = K_layer.Reshape((self.partial_n, self.latent_dim/2))
		encode_1 = RNN_UNIT(self.latent_dim/2)
		encode_2 = RNN_UNIT(self.latent_dim, return_sequences=True)

		def encode_partials(seq):
			encoded = [None]*self.partial_n
			for i in range(self.partial_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.partial_ts, self.input_dim))(seq)
				encoded[i] = encode_1(rs)
			return encode_reshape(K_layer.concatenate(encoded, axis=1))

		encoded = encode_partials(reshaped)
		encoded = encode_2(encoded)

		z = K_layer.Input(shape=(self.latent_dim,))
		decoder_activation = 'tanh'
		decode_expmap_1 = K_layer.Dense(self.latent_dim/2, activation=decoder_activation)
		decode_expmap_2 = K_layer.Dense(self.output_dim, activation=decoder_activation)

		#def angle_relu(x):
		#	return tf.maximum(tf.minimum(x, 1.0), -1.0)

		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_residual_1 = RNN_UNIT(self.latent_dim/2, return_sequences=True, activation=decoder_activation)
		decode_residual_2 = RNN_UNIT(self.output_dim, return_sequences=True, activation=decoder_activation)

		#decode_activate = K_layer.Lambda(lambda x: K.tanh(x), output_shape=(self.timesteps, self.output_dim))
		#decode_cos = K_layer.Lambda(lambda x: K.sign(x[1])*K.sqrt(1 - K.square(x[0])), output_shape=(self.timesteps, self.output_dim))
		#decode_atan2 = K_layer.Lambda(lambda x: tf.atan2(x[0], x[1]), output_shape=(self.timesteps, self.output_dim))

		def decode_angle(e):
			angle_sin = decode_expmap_2(decode_expmap_1(e))
			#angle_cos = decode_expmap_2(e)

			residual = decode_repete(e)
			residual = decode_residual_2(decode_residual_1(residual))
			angle_sin = K_layer.Activation(decoder_activation)(K_layer.add([decode_repete(angle_sin), residual]))
			#angle_cos = decode_activate(K_layer.add([decode_repete(angle_cos), residual]))

			#angle_sin = K_layer.Activation(angle_relu)(angle_sin)
			#angle_cos = decode_cos([angle_sin, angle_cos])
			#angle = decode_atan2([angle_sin, angle_cos])
			return angle_sin


		angles = [None]*len(self.hierarchies_index)
		for i,k in enumerate(self.hierarchies_index):
			e = K_layer.Lambda(lambda x: x[:,k], output_shape=(self.latent_dim,))(encoded)
			angles[i] = decode_angle(e)

		decoded =  K_layer.concatenate(angles, axis=1)
		decoded_ = decode_angle(z)

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.autoencoder = Model(inputs, decoded)

		def mse(yTrue, yPred):
		# 	yt = K.reshape(yTrue, (-1, self.timesteps, self.output_dim))
		#  	yp = K.reshape(yPred, (-1, self.timesteps, self.output_dim))
			a = yTrue
			b = yPred
			return tf.reduce_mean(tf.abs(tf.atan2(tf.sin(a - b), tf.cos(a - b))))
		 	#loss = K.square(K.sin(yTrue) - K.sin(yPred))
		 	#loss = loss + K.square(K.cos(yTrue) - K.cos(yPred))
		 	#loss = K.mean(K.sqrt(loss))
		 	#return loss

		# opt = optimizers.Nadam(lr=L_RATE)
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

	def __alter_y(self, y):
		if len(self.hierarchies) == 1:
			return y
		y = np.repeat(y, len(self.hierarchies), axis=0)
		y = np.reshape(y, (-1, len(self.hierarchies), self.timesteps, y.shape[-1]))
		for i, h in enumerate(self.hierarchies):
			for j in range(h+1, self.timesteps):
				y[:,i,j] = y[:,i,h]
		return np.reshape(y, (-1, self.timesteps*len(self.hierarchies), y.shape[-1]))


	def __alter_parameterization(self, y):
		norm_y = self.normalize_angle(y)
		return y, norm_y

	def to_euler(self, x):
		x = self.recover(x)
		b, t, d = x.shape
		x = np.reshape(x, (-1,3))
		euler_x = np.zeros(x.shape)
		for i in range(x.shape[0]):
			euler_x[i] = rotmat2euler(expmap2rotmat(x[i]))
		return np.reshape(euler_x, (b,t,d))[:,:,EULER_IDEX]

	def expmap_error(self, yTrue, yPred):
		return np.mean(np.sqrt(np.sum(np.square(yTrue - yPred), -1)), 0)

	def euler_error(self, yTrue, yPred):
		# from Matinez
		return np.mean(np.sqrt(np.sum(np.square(self.to_euler(yTrue) - self.to_euler(yPred)), -1)), 0)

	def load_validation_data(self, load_path):
		y = [None]*15
		i = 0
		cut = (self.predict_hierarchies[0]+1)*self.partial_ts
		for basename in metric_baselines.iter_actions():
			cond = np.load(load_path + 'euler/' + basename + '_cond.npy')[:,-self.timesteps:]
			y[i] = np.zeros(cond.shape)
			y[i][:,:cut] = cond[:,-cut:]
			y[i][:,cut:] = np.load(load_path + 'euler/' + basename + '_gt.npy')[:,:self.timesteps-cut]
			i += 1
		y = np.concatenate(y, axis=0)
		print y.shape
		y, x = self.__alter_parameterization(y)
		return y, x


	def validate_autoencoding(self, test_data_x, test_data_y):
		y_test_pred = self.encoder.predict(test_data_x)[:,-1]
		y_test_pred = self.decoder.predict(y_test_pred)
		y_test_pred = self.unormalize_angle(y_test_pred)
		return self.expmap_error(test_data_y, y_test_pred)

	def validate_prediction(self, x_train, x_test, y_test):
		e = self.encoder.predict(x_train)
		embedding = e[:, self.predict_hierarchies]
		mean_diff, diff = metrics.get_embedding_diffs(embedding[:,1], embedding[:,0])
		add_std = np.std(diff, axis=0)
		z = self.encoder.predict(x_test)[:,self.predict_hierarchies[0]]
		z = z + mean_diff
		y_test_pred = self.decoder.predict(z)
		y_test_pred = self.unormalize_angle(y_test_pred)
		return add_std, self.euler_error(y_test, y_test_pred)


	def run(self, data_iterator, valid_data):
		load_path = '../human_motion_pred/baselines/'
		print self.loss_opt_str
		if not self.load():
			#test_data_y, test_data_x = self.load_validation_data(load_path)

			# from keras.utils import plot_model
			# plot_model(self.autoencoder, to_file='model.png')
			loss = 10000
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, _ in iter1:
					#image.plot_fk_from_euler(x[:3], title='test')
					y, x = self.__alter_parameterization(x)
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
						#self.autoencoder.save_weights(self.save_path, overwrite=True)
						loss = new_loss
						print 'Saved model - ', loss

					rand_idx = np.random.choice(x.shape[0], 100, replace=False)
					y_gt = y[rand_idx]
					mse = self.validate_autoencoding(x[rand_idx], y_gt)
					#mse_test = self.validate_autoencoding(test_data_x, test_data_y)
					#add_std, mse_pred = self.validate_prediction(x[rand_idx], test_data_x, test_data_y)

					#add_mean_std, add_std_std = np.mean(add_std), np.std(add_std)
					#print 'STD', add_mean_std, add_std_std
					print 'MSE', np.mean(mse)
					#print 'MSE TEST', np.mean(mse_test)
					#print 'MSE PRED', mse_pred[[-9,-7,-3,-1]]

					#with open('../new_out/%s_t%d_l%d_%s_log.csv'%(NAME, self.timesteps, self.latent_dim, self.loss_opt_str), 'a+') as f:
					#	spamwriter = csv.writer(f)
					#	spamwriter.writerow([new_loss, mse, mse_test, mse_pred, add_mean_std, add_std_std, self.loss_opt_str])

				iter1, iter2 = tee(iter2)

			data_iterator = iter2
		else:
			# load embedding
			embedding = []
			i = 0
			for x,y in data_iterator:
				y_gt, norm_y = self.__alter_parameterization(x)
				e = self.encoder.predict(norm_y)
				#y_test_pred = self.decoder.predict(e[:,-1])
				#y_test_pred = self.unormalize_angle(y_test_pred)
				#print self.euler_error(y_gt, y_test_pred)
				#np.save('../data/embedding/t40-l1024-euler-nadam-meanAbs-lr0.0001/emb_%d.npy'%i, e[:,self.predict_hierarchies])
				#i += 1
				#print i
				#continue
				if len(embedding) == 0:
					embedding = e[:, self.predict_hierarchies]
				else:
					embedding = np.concatenate((embedding, e[:,self.predict_hierarchies]), axis=0)
				#break
			#return
			embedding = np.array(embedding)
			print 'emb', embedding.shape
			mean_diff, diff = metrics.get_embedding_diffs(embedding[:,1], embedding[:,0])
			print 'std', np.std(diff)

			_N = 8
			#methods = ['closest', 'closest_partial', 'mean-5', 'add', 'fn']
			methods = ['0v']
			nn = NN.Forward_NN({'input_dim':self.latent_dim, 'output_dim':self.latent_dim, 'mode':'sample'})
			nn.run(None)
			#nn = None

			cut_e = self.predict_hierarchies[0]
			cut_x = self.hierarchies[0]+1
			pred_n = self.hierarchies[1]-cut_x+1
			error = {act: {m: {'euler': None} for m in methods} for act in metric_baselines.iter_actions()}

			# a_n = 0
			for basename in metric_baselines.iter_actions():
				print basename, '================='

				# x, y = valid_data
				# x = np.zeros((_N, self.timesteps, 96))
				# x[:,:cut_x+1] = np.load(load_path + 'xyz/' + basename + '_cond.npy')[:,-cut_x-1:]

				y = np.zeros((_N, self.timesteps, 99))
				y[:,:cut_x] = np.load(load_path + 'euler/' + basename + '_cond.npy')[:,-cut_x:]
				gtp_y = np.load(load_path + 'euler/' + basename + '_gt.npy')[:,:pred_n]

				y, norm_y = self.__alter_parameterization(y)
				enc = self.encoder.predict(norm_y)
				partial_enc = enc[:,cut_e]

				fn_pred = nn.model.predict(partial_enc)
				#poses = [None]*(len(methods)+2)
				#poses[1] = self.recover(gtp_y[0:1])
				#poses[0] = gtp_y[0:1]

				for k,method in enumerate(methods):
					new_enc = np.zeros(partial_enc.shape)
					for i in tqdm(range(_N)):
						if method == 'closest_partial':
							idx = metrics.closest_partial_index(embedding[:,0], partial_enc[i])
							new_enc[i] = embedding[idx,1]
						elif 'mean' in method:
							n = int(method.split('-')[1])
							new_enc[i] = metrics.closest_mean(embedding[:,1], partial_enc[i], n=n)
						elif method == 'add':
							new_enc[i] = partial_enc[i]+mean_diff
						elif method == 'closest':
							new_enc[i] = metrics.closest(embedding[:,1], partial_enc[i])
						elif method == 'fn':
							new_enc[i] = fn_pred[i]

					model_pred = None
					if method == '0v':
						model_pred = np.zeros(gtp_y.shape)
						for i in range(pred_n):
							model_pred[:,i] = y[:,cut_x-1]
					else:
						model_pred = self.decoder.predict(new_enc)[:,cut_x:]
						model_pred = self.unormalize_angle(model_pred)

					error[basename][method]['euler'] = self.euler_error(gtp_y, model_pred) #y[:, -pred_n:], model_pred)
					print method
					print error[basename][method]['euler'][[1,3,7,9]]
					error[basename][method]['euler'] = error[basename][method]['euler'].tolist()

					#poses[k+2] = self.recover(model_pred[:1])

					# error[method]['z'] = np.mean([np.linalg.norm(new_enc[i] - enc[i,-1]) for i in range(_N)])
					# print error[method]['z']

					# for i in range(_N):
					# 	pose_err = metrics.pose_seq_error(gtp_x[i], model_pred[i,:,:self.euler_start], cumulative=True)
					# 	error[method]['pose'] = error[method]['pose'] + np.array(pose_err)
					# error[method]['pose'] = error[method]['pose']/_N
					# print error[method]['pose']
					# error[method]['pose'] = error[method]['pose'].tolist()

				#poses = np.concatenate(poses, axis=0)
				#image.plot_fk_from_euler(poses, title='%s_gt_agt_%s'%(basename, score_name), image_dir='../new_out/')

			with open('../new_out/zero_velocity_validation-testset-mseMartinez.json', 'wb') as result_file:
                               json.dump(error, result_file)

			#with open('../new_out/%s_t%d_l%d_opt-%s_validation-testset-mseMartinez.json'%(NAME, self.timesteps, self.latent_dim, self.loss_opt_str), 'wb') as result_file:
			#	json.dump(error, result_file)



if __name__ == '__main__':
	data_iterator, valid_data, config = parser.get_parse(NAME)
	ae = H_expmap_RNN_R(config)
	ae.run(data_iterator, valid_data)

