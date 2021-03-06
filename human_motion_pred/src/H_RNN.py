import matplotlib
matplotlib.use('Agg')

import numpy as np
from keras.layers import Input, RepeatVector, Lambda, concatenate
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K

import seq2seq_model__
from utils import parser
from utils import translate__

MODEL_NAME = 'H_GRU'
USE_GRU = True
HAS_LABELS = False
LOSS = 'mean_squared_error'

if USE_GRU:
	from keras.layers import GRU as RNN_UNIT
else:
	from keras.layers import LSTM as RNN_UNIT
	MODEL_NAME = 'H_LSTM'

class H_RNN(seq2seq_model__.seq2seq_ae__):

	def make_model(self):
		inputs = Input(shape=(self.timesteps, self.data_dim))
		encoded = RNN_UNIT(self.latent_dim, return_sequences=True, activation='tanh')(inputs)

		z = Input(shape=(self.latent_dim,))
		decode_1 = RepeatVector(self.timesteps)
		decode_2 = RNN_UNIT(self.data_dim, return_sequences=True, activation='linear')

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

		def customLoss(yTrue, yPred):
			return K.mean(K.sqrt(K.square(K.sin(yTrue) - K.sin(yPred)) + K.square(K.cos(yTrue) - K.cos(yPred))))

		self.loss = customLoss

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.autoencoder = Model(inputs, decoded)
		self.recompile_opt()

		#self.autoencoder.summary()
		#self.encoder.summary()
		#self.decoder.summary()

	def recompile_opt(self):
		opt = RMSprop(lr=self.lr)
		self.autoencoder.compile(optimizer=opt, loss=self.loss)

	# def run(self, data_iterator):
	# 	self.load()
	# 	if not self.trained:
	# 		# from keras.utils import plot_model
	# 		# plot_model(self.autoencoder, to_file='model.png')
	# 		for x in data_iterator:
	# 			# y = np.copy(x)
	# 			x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, x, test_size=self.cv_splits)
	# 			y_train = self.alter_y(y_train)
	# 			y_test = self.alter_y(y_test)
	# 			print x_train.shape, x_test.shape, y_train.shape, y_test.shape
	# 			from utils import image
	# 			image.plot_data(x_train[0])
 #                #xyz = translate__.batch_expmap2xyz(y_train[:5,:5], self)
 #                #image.plot_poses(xyz)

	# 			history = self.autoencoder.fit(x_train, y_train,
	# 						shuffle=True,
	# 						epochs=self.epochs,
	# 						batch_size=self.batch_size,
	# 						validation_data=(x_test, y_test))

	# 			self.post_train_step(history.history['loss'][0], x_test, [OPT, LOSS])

if __name__ == '__main__':
	train_set_gen, test_set, config = parser.get_parse(MODEL_NAME, HAS_LABELS)
	ae = H_RNN(config, HAS_LABELS)
	ae.run(train_set_gen, test_set, HAS_LABELS)
