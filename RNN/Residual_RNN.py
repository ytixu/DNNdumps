import matplotlib
matplotlib.use('Agg')

import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, RepeatVector, Lambda, concatenate, Dense, Add
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K

from utils import parser, image, embedding_plotter, recorder, metrics, metric_baselines, association_evaluation
from Forward import NN

LEARNING_RATE = 0.00005
NAME = 'R_LSTM'
USE_GRU = True
if USE_GRU:
	from keras.layers import GRU
	NAME = 'R_GRU'
else:
	from keras.layers import LSTM

class R_LSTM:
	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		self.decoder = None

		self.epochs = args['epochs']
		self.batch_size = args['batch_size']
		print 'batch size -------------- %d' % (self.batch_size)
		self.periods = args['periods'] if 'periods' in args else 10
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 5
		self.label_dim = args['label_dim']
		self.labels = args['labels']
		print self.labels, self.label_dim
		self.input_dim = args['input_dim'] + self.label_dim
		self.output_dim = args['output_dim'] + self.label_dim
		self.motion_dim = args['input_dim']
		self.hierarchies = args['hierarchies'] if 'hierarchies' in args else [9,14,24]
		self.latent_dim = args['latent_dim'] if 'latent_dim' in args else (args['input_dim']+args['output_dim'])/2
		self.trained = args['mode'] == 'sample' if 'mode' in args else False
		self.load_path = args['load_path']
		self.save_path = args['save_path']
		self.log_path = args['log_path']

		self.joint_number = args['input_dim']/3

		self.MODEL_CODE = metrics.L_LSTM
		self.NAME = NAME
		# self.history = recorder.LossHistory()

	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.input_dim))
		encoded = GRU(self.latent_dim, return_sequences=True)(inputs)

		z = Input(shape=(self.latent_dim,))
		decode_pose = Dense(self.motion_dim, activation='tanh')
		decode_name = Dense(self.label_dim, activation='relu')
		decode_repete = RepeatVector(self.timesteps)
		decode_residual = GRU(self.output_dim, return_sequences=True)

		decoded = [None]*len(self.hierarchies)
		residual = [None]*len(self.hierarchies)
		for i, h in enumerate(self.hierarchies):
			e = Lambda(lambda x: x[:,h], output_shape=(self.latent_dim,))(encoded)
			decoded[i] = concatenate([decode_pose(e), decode_name(e)], axis=1)
			residual[i] = decode_repete(e)
			residual[i] = decode_residual(residual[i])
			decoded[i] = Add()([decode_repete(decoded[i]), residual[i]])

		decoded = concatenate(decoded, axis=1)
		residual = concatenate(residual, axis=1)

		decoded_ = concatenate([decode_pose(z), decode_name(z)], axis=1)
		residual_ = decode_repete(z)
		residual_ = decode_residual(residual_)
		decoded_ = Add()([decode_repete(decoded_), residual_])

		def customLoss(yTrue, yPred):
			yt = K.reshape(yTrue[:,:,-self.label_dim:], (-1, self.timesteps, self.timesteps, self.label_dim))
			yp = K.reshape(yPred[:,:,-self.label_dim:], (-1, self.timesteps, self.timesteps, self.label_dim))
			loss = 0
			yTrue = K.reshape(yTrue[:,:,:-self.label_dim], (-1, self.timesteps, self.timesteps, self.timesteps/3, 3))
			yPred = K.reshape(yPred[:,:,:-self.label_dim], (-1, self.timesteps, self.timesteps, self.timesteps/3, 3))
			# loss += K.mean(K.sqrt(K.sum(K.square(yTrue-yPred), axis=-1)))
			# loss += K.mean(K.sqrt(K.sum(K.square(yt - yp), axis=-1)))/self.timesteps
			loss += K.mean(K.sqrt(K.sum(K.square(yTrue-yPred), axis=-1))) + K.mean(K.abs(yt-yp))/self.timesteps
			return loss

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.autoencoder = Model(inputs, decoded)
		opt = RMSprop(lr=LEARNING_RATE)
		self.autoencoder.compile(optimizer=opt, loss=customLoss)

		self.autoencoder.summary()
		self.encoder.summary()
		self.decoder.summary()


	def load(self):
		self.make_model()
		if self.trained:
			self.autoencoder.load_weights(self.load_path)
			print 'loaded'
			return True
		return False

	#def __alter_y(self, y):
	#	# return y
	#	new_y = [None]*len(self.hierarchies)
	#	for i, h in enumerate(self.hierarchies):
	#		new_y[i] = np.copy(y)
	#	return np.concatenate(new_y, axis=1)

	def __alter_y(self, y):
		y = np.repeat(y, len(self.hierarchies), axis=0)
		y = np.reshape(y, (-1, len(self.hierarchies), self.timesteps, y.shape[-1]))
		for i, h in enumerate(self.hierarchies):
			for j in range(h+1, self.timesteps):
				y[:,i,j] = y[:,i,h]
		print y.shape
		return np.reshape(y, (-1, self.timesteps*len(self.hierarchies), y.shape[-1]))

	def __alter_label(self, x, y):
		idx = np.random.choice(x.shape[0], x.shape[0]/2)
		x[idx,:,-self.label_dim:] = 0
		y[idx,:,-self.label_dim:] = 0
		return x, y

	def run(self, data_iterator, valid_data):
		model_vars = [NAME, self.latent_dim, self.timesteps, self.batch_size]
		if not self.load():
			# from keras.utils import plot_model
			# plot_model(self.autoencoder, to_file='model.png')
			loss = 10000
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					x_data, y_data = self.__alter_label(x, y)
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=self.cv_splits)
					y_train = self.__alter_y(y_train)
					y_test = self.__alter_y(y_test)
					# print np.sum(y_train[:,0,-self.label_dim:], axis=0)
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
						#y_test_decoded = self.autoencoder.predict(x_test[:1])
						#y_test_decoded = np.reshape(y_test_decoded, (len(self.hierarchies), self.timesteps, -1))
						#image.plot_poses(x_test[:1,:,:-self.label_dim], y_test_decoded[:,:,:-self.label_dim])
						# image.plot_hierarchies(y_test_orig[:,:,:-self.label_dim], y_test_decoded[:,:,:-self.label_dim])
						self.autoencoder.save_weights(self.save_path, overwrite=True)
						rand_idx = np.random.choice(x_test.shape[0], 25, replace=False)
						metrics.validate(x_test[rand_idx], self, '../new_out/validate-residual.txt')

					del x_train, x_test, y_train, y_test
				iter1, iter2 = tee(iter2)

			# self.history.record(self.log_path, model_vars)
			data_iterator = iter2

		# metric_baselines.compare(self)
		# metrics.gen_long_sequence(valid_data, self)

		# embedding_plotter.see_hierarchical_embedding(self.encoder, self.decoder, data_iterator, valid_data, model_vars, self.label_dim)
		# iter1, iter2 = tee(data_iterator)
		# metrics.validate(valid_data, self)

		nn = NN.Forward_NN({'input_dim':self.latent_dim, 'output_dim':self.latent_dim, 'mode':'sample'})
		nn.run(None)
		metrics.plot_metrics(self, data_iterator, valid_data, nn)
		# association_evaluation.plot_best_distance_function(self, valid_data, data_iterator)
		# association_evaluation.eval_generation(self, valid_data, data_iterator)
		# association_evaluation.eval_center(self, valid_data, 'sitting')
		# association_evaluation.transfer_motion(self, valid_data, 'sitting', 'walking', data_iterator)
		# association_evaluation.plot_transfer_motion(self, '../new_out/transfer_motion-sitting-to-greeting-scores.npy')
		# association_evaluation.plot_transfer_motion(self, '../new_out/transfer_motion-sitting-to-walking-scores.npy')
		# association_evaluation.eval_generation_from_label(self, data_iterator)
		# association_evaluation.plot_add(self, data_iterator)
		# metrics.plot_metrics_labels(self, data_iterator, valid_data)
		# metric_baselines.compare_label_embedding(self, nn, data_iterator)
		# association_evaluation.eval_distance(self, valid_data)
		# evaluate.eval_pattern_reconstruction(self.encoder, self.decoder, iter2)

if __name__ == '__main__':
	data_iterator, valid_data, config = parser.get_parse(NAME, labels=True)
	ae = R_LSTM(config)
	ae.run(data_iterator, valid_data)

