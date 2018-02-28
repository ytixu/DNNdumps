import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, LSTM, RepeatVector, Flatten, Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K

from utils import parser, image, embedding_plotter, recorder, option_visualizer


NAME = 'Option_LSTM'

class Option_LSTM:

	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		# self.decoder = None

		self.epochs = args['epochs']
		self.batch_size = args['batch_size']
		self.periods = args['periods'] if 'periods' in args else 15
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 5
		self.input_dim = args['input_dim']
		self.output_dim = args['output_dim']
		self.latent_dim = args['latent_dim'] if 'latent_dim' in args else (args['input_dim']+args['output_dim'])/2
		self.option_dim = args['option_dim'] if 'option_dim' in args else 10
		self.trained = args['mode'] == 'sample' if 'mode' in args else False
		self.load_path = args['load_path']
		self.log_path = args['log_path']
		self.out_dim = self.output_dim*self.timesteps
		self.alpha = 10

		self.history = recorder.LossHistory()

	def make_model(self):	
		inputs = Input(shape=(self.timesteps, self.input_dim))
		encoded = LSTM(self.latent_dim)(inputs)
		
		optioned = RepeatVector(self.option_dim)(encoded)
		optioned = Flatten()(optioned)
		
		decoded = RepeatVector(self.timesteps)(optioned)
		decoded = LSTM(self.output_dim*self.option_dim, return_sequences=True)(decoded)
		decoded = Flatten()(decoded)

		class_layer = Dense(self.option_dim, activation='softmax')(encoded)
		merged = concatenate([decoded, class_layer])

		self.encoder = Model(inputs, encoded)
		self.autoencoder = Model(inputs, merged)
		
		def min_loss(y_true, y_pred):
			class_vector = y_pred[:, -self.option_dim:]
			prediction = y_pred[:, :-self.option_dim]
			ry_pred = K.reshape(prediction, [-1, self.option_dim, self.out_dim])
			ry_true = K.tile(y_true, (1, self.option_dim))
			ry_true = K.reshape(ry_true, [-1, self.option_dim, self.out_dim])
			sum_sqr_ = K.sum(K.square(ry_pred - ry_true), axis=2)
			min_ = K.min(sum_sqr_, axis=1)
			class_loss = K.dot(class_vector, K.transpose(sum_sqr_))
			return self.alpha*min_ + class_loss

		self.autoencoder.compile(optimizer='RMSprop', loss=min_loss)

		self.autoencoder.summary()
		self.encoder.summary()

	def _match(self, y_true, y_pred):
		class_vector = y_pred[:, -self.option_dim:]
		prediction = y_pred[:, :-self.option_dim]
		ry_pred = np.reshape(prediction, [-1, self.option_dim, self.out_dim])
		ry_true = np.tile(y_true, (1,self.option_dim))
		ry_true = np.reshape(ry_true, [-1, self.option_dim, self.out_dim])
		min_loss_idx = np.argmin(np.sum(np.square(ry_pred - ry_true), axis=2), axis=1)
		return min_loss_idx, np.reshape(ry_pred[:,min_loss_idx], [-1, self.timesteps, self.output_dim])

	def best(self, y_true, y_pred):
		return self._match(y_true, y_pred)[1]

	def best_option(self, y_true, y_pred):
		return self._match(y_true, y_pred)[0]

	def run(self, data_iterator): 
		self.make_model()
		model_vars = [NAME, self.latent_dim, self.timesteps, self.batch_size, self.option_dim]

		if self.trained:
			self.autoencoder.load_weights(self.load_path)
		else:
			iter1, iter2 = tee(data_iterator)
			for i in range(self.periods):
				for x, y in iter1:
					x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
					y_train = np.reshape(y_train, (-1, self.out_dim))
					y_test = np.reshape(y_test, (-1, self.out_dim))
					self.autoencoder.fit(x_train, y_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=(x_test, y_test),
								callbacks=[self.history])

					y_test_decoded = self.autoencoder.predict(x_test[:1])
					y_test_sample = np.reshape(y_test[:1], (-1, self.timesteps, self.output_dim))
					# y_test_decoded = self.autoencoder.predict(x_test[:2])
					# y_test_sample = np.reshape(y_test[:2], (-1, self.timesteps, self.output_dim))
					# image.plot_batch_2D(y_test_sample, self.best(y_test[:1], y_test_decoded))

					class_vector = y_test_decoded[:, -self.option_dim:]
					prediction = np.reshape(y_test_decoded[:, :-self.option_dim], [-1, self.option_dim, self.timesteps, self.output_dim])
					best_opt = self.best_option(y_test[:1], y_test_decoded)
					predict_best_opt = np.argmax(class_vector, axis=1)
					image.plot_batch_2D(x_test[:1], y_test_sample, prediction, best_opt, predict_best_opt)

					# image.plot_graph(x_test[:2], y_test_sample, prediction, best_opt, predict_best_opt)
				
				# self.autoencoder.save_weights(self.load_path, overwrite=True)
				iter1, iter2 = tee(iter2)

			# data_iterator = iter2
			# self.history.record(self.log_path, model_vars)

		# iter1, iter2 = tee(data_iterator)
		# embedding_plotter.see_embedding(self.encoder, iter1, model_vars)

		# for x,y in iter2:
		# 	random_idx = np.random.choice(len(x))
		# 	print x[random_idx:random_idx+1]
		# 	y_test_decoded = self.autoencoder.predict(x[random_idx:random_idx+1])
		# 	y_test_decoded = np.reshape(y_test_decoded, [-1, self.option_dim, self.timesteps, self.output_dim])
		# 	image.plot_options(y[random_idx:random_idx+1], y_test_decoded)

		# for x, y in iter2:
		# 	y_decoded = self.autoencoder.predict(x)
		# 	y_encoded = self.encoder.predict(x)
		# 	opt = self.best_option(y, y_decoded).flatten()
		# 	option_visualizer.visualize(y, y_encoded, y_decoded, opt)

if __name__ == '__main__':
	data_iterator, config = parser.get_parse(NAME)
	ae = Option_LSTM(config)
	ae.run(data_iterator)
