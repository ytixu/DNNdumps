import matplotlib
matplotlib.use('Agg')

import numpy as np
from keras.layers import Input, RepeatVector, Lambda, concatenate, Dense, Add
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K

import seq2seq_model__
from utils import parser
from utils import translate__

MODEL_NAME = 'R_GRU'
USE_GRU = True
HAS_LABELS = False #True

if USE_GRU:
	from keras.layers import GRU as RNN_UNIT
else:
	from keras.layers import LSTM as RNN_UNIT
	MODEL_NAME = 'R_LSTM'

class R_RNN(seq2seq_model__.seq2seq_ae__):

	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.data_dim))
		encoded = RNN_UNIT(self.latent_dim, return_sequences=True, activation='tanh')(inputs)

		z = Input(shape=(self.latent_dim,))
		decode_pose_1 = Dense(self.latent_dim/2) # self.motion_dim)
		decode_pose_2 = Dense(self.data_dim)
		# decode_name = Dense(self.label_dim, activation='relu')
		decode_repete = RepeatVector(self.timesteps)
		decode_residual = RNN_UNIT(self.data_dim, return_sequences=True, activation='linear')
		decode_add = Add()

		decoded = [None]*len(self.hierarchies)
		residual = [None]*len(self.hierarchies)
		for i, h in enumerate(self.hierarchies):
			e = Lambda(lambda x: x[:,h], output_shape=(self.latent_dim,))(encoded)
			# decoded[i] = concatenate([decode_pose(e), decode_name(e)], axis=1)
			decoded[i] = decode_pose_2(decode_pose_1(e))
			residual[i] = decode_repete(e)
			residual[i] = decode_residual(residual[i])
			decoded[i] = decode_add([decode_repete(decoded[i]), residual[i]])

		decoded = concatenate(decoded, axis=1)
		residual = concatenate(residual, axis=1)

		# decoded_ = concatenate([decode_pose(z), decode_name(z)], axis=1)
		decoded_ = decode_pose_2(decode_pose_1(z))
		residual_ = decode_repete(z)
		residual_ = decode_residual(residual_)
		decoded_ = decode_add([decode_repete(decoded_), residual_])

		# def customLoss(yTrue, yPred):
		# 	yt = K.reshape(yTrue[:,:,-self.label_dim:], (-1, len(self.hierarchies), self.timesteps, self.label_dim))
		# 	yp = K.reshape(yPred[:,:,-self.label_dim:], (-1, len(self.hierarchies), self.timesteps, self.label_dim))
		# 	loss = 0
		# 	print K.int_shape(yTrue), K.int_shape(yPred)
		# 	yTrue = K.reshape(yTrue[:,:,:-self.label_dim], (-1, len(self.hierarchies), self.timesteps, self.motion_dim/3, 3))
		# 	yPred = K.reshape(yPred[:,:,:-self.label_dim], (-1, len(self.hierarchies), self.timesteps, self.motion_dim/3, 3))
		# 	# loss += K.mean(K.sqrt(K.sum(K.square(yTrue-yPred), axis=-1)))
		# 	# loss += K.mean(K.sqrt(K.sum(K.square(yt - yp), axis=-1)))/self.timesteps
		# 	loss += K.mean(K.sqrt(K.sum(K.square(yTrue-yPred), axis=-1))) + K.mean(K.abs(yt-yp))/len(self.hierarchies)
		# 	return loss

		self.loss = 'mean_squared_error'

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.autoencoder = Model(inputs, decoded)
		self.recompile_opt()

		self.autoencoder.summary()
		self.encoder.summary()
		self.decoder.summary()

	def recompile_opt(self):
		opt = RMSprop(lr=self.lr)
		self.autoencoder.compile(optimizer=opt, loss=self.loss)

	# def run(self, data_iterator):
	# 	self.load()
	# 	if not self.trained:
	# 		# from keras.utils import plot_model
	# 		# plot_model(self.autoencoder, to_file='model.png')
	# 		for x in data_iterator:
	# 			x_data = self.alter_label(x)
	# 			x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_data, x_data, test_size=self.cv_splits)
	# 			y_train = self.alter_y(y_train)
	# 			y_test = self.alter_y(y_test)
	# 			print x_train.shape, x_test.shape, y_train.shape, y_test.shape
	# 			#from utils import image
 #                #xyz = translate__.batch_expmap2xyz(y_train[:5,:5], self)
 #                #image.plot_poses(xyz)

	# 			history = self.autoencoder.fit(x_train, y_train,
	# 						shuffle=True,
	# 						epochs=self.epochs,
	# 						batch_size=self.batch_size,
	# 						validation_data=(x_test, y_test))

	# 			self.post_train_step(history.history['loss'][0], x_test)

if __name__ == '__main__':
	train_set_gen, test_set, config = parser.get_parse(MODEL_NAME, HAS_LABELS)
	ae = R_RNN(config, HAS_LABELS)
	ae.run(train_set_gen, test_set, HAS_LABELS)
