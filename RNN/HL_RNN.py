import numpy as np
from itertools import tee
from sklearn import cross_validation
from keras.layers import Input, RepeatVector, Lambda, concatenate
from keras.models import Model

from utils import parser, image, embedding_plotter, recorder, metrics, metric_baselines

NAME = 'HL_LSTM'
USE_GRU = True
if USE_GRU:
	from keras.layers import GRU
	NAME = 'HL_GRU'
else:
	from keras.layers import LSTM

class HL_LSTM:
	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		self.decoder = None

		self.epochs = args['epochs']
		self.batch_size = args['batch_size']
		self.periods = args['periods'] if 'periods' in args else 10
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2

		self.timesteps = args['timesteps'] if 'timesteps' in args else 5
		self.label_dim = args['label_dim']
		self.labels = args['labels']
		self.input_dim = args['input_dim'] + self.label_dim
		self.output_dim = args['output_dim'] + self.label_dim
		self.hierarchies = args['hierarchies'] if 'hierarchies' in args else [int(self.timesteps*i/4.0-1) for i in range(1,5)]
		print self.hierarchies
		self.latent_dim = args['latent_dim'] if 'latent_dim' in args else (args['input_dim']+args['output_dim'])/2
		self.trained = args['mode'] == 'sample' if 'mode' in args else False
		self.load_path = args['load_path']
		self.log_path = args['log_path']

		self.MODEL_CODE = metrics.HL_LSTM
		# self.history = recorder.LossHistory()

	def make_model(self):
		inputs_1 = Input(shape=(self.timesteps, self.input_dim))
		inputs_2 = Input(shape=(self.input_dim, ))

		encoded = None
		if USE_GRU:
			encoded = GRU(self.latent_dim, return_sequences=True)(inputs_1)
		else:		
			encoded = LSTM(self.latent_dim, return_sequences=True)(inputs_1)

		z = Input(shape=(self.latent_dim,))
		decode_1 = RepeatVector(self.timesteps)
		decode_2 = None 
		if USE_GRU:
			decode_2 = GRU(self.output_dim, return_sequences=True)
		else:
			decode_2 = LSTM(self.output_dim, return_sequences=True)

		decoded = [None]*len(self.hierarchies)
		for i, h in enumerate(self.hierarchies):
			e = Lambda(lambda x: x[:,h], output_shape=(self.latent_dim,))(encoded)
			decoded[i] = decode_1(concatenate([e, inputs_2], axis=1))
			decoded[i] = decode_2(decoded[i])
		decoded = concatenate(decoded, axis=1)

		decoded_ = decode_1(concatenate([z, inputs_2], axis=1))
		decoded_ = decode_2(decoded_)
		
		self.encoder = Model(inputs_1, encoded)
		self.decoder = Model([z, inputs_2], decoded_)
		self.autoencoder = Model([inputs_1, inputs_2], decoded)
		self.autoencoder.compile(optimizer='RMSprop', loss='mean_squared_error')

		self.autoencoder.summary()
		self.encoder.summary()
		self.decoder.summary()

	def load(self):
		self.make_model()
		if self.trained:
			self.autoencoder.load_weights(self.load_path)
			return True
		return False

		
	def __alter_y(self, y):
		y = np.repeat(y, len(self.hierarchies), axis=0)
		y = np.reshape(y, (-1, len(self.hierarchies), self.timesteps, y.shape[-1]))
		for i, h in enumerate(self.hierarchies):
			y[:,i,h+1:,:-self.label_dim] = 0.0
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
					y_test_orig = np.copy(y_test[:1])
					y_test = self.__alter_y(y_test)
					# print np.sum(y_train[:,0,-self.label_dim:], axis=0)
					history = self.autoencoder.fit([x_train, x_train[:,0]], y_train,
								shuffle=True,
								epochs=self.epochs,
								batch_size=self.batch_size,
								validation_data=([x_test, x_test[:,0]], y_test))
								# callbacks=[self.history])

					new_loss = np.mean(history.history['loss'])
					if new_loss < loss:
						print 'Saved model - ', loss
						loss = new_loss
						y_test_decoded = self.autoencoder.predict([x_test[:1], x_test[:1,0]])
						print np.linalg.norm(y_test_decoded[0,0,-self.label_dim:] - y_test_orig[0,0,-self.label_dim:])
						image.plot_poses(y_test_orig[:,:,:-self.label_dim], y_test_decoded[:,:,:-self.label_dim])
						# image.plot_hierarchies(y_test_orig[:,:,:-self.label_dim], y_test_decoded[:,:,:-self.label_dim])
						self.autoencoder.save_weights(self.load_path, overwrite=True)

				iter1, iter2 = tee(iter2)
			
			# self.history.record(self.log_path, model_vars)
			data_iterator = iter2
		
		# metric_baselines.compare(self)
		# metrics.gen_long_sequence(valid_data, self)

		# embedding_plotter.see_hierarchical_embedding(self.encoder, self.decoder, data_iterator, valid_data, model_vars, self.label_dim)
		# iter1, iter2 = tee(data_iterator)
		metrics.validate(valid_data, self.encoder, self.decoder, self.timesteps, metrics.HL_LSTM)
		# evaluate.eval_pattern_reconstruction(self.encoder, self.decoder, iter2)

if __name__ == '__main__':
	data_iterator, valid_data, config = parser.get_parse(NAME, labels=True)
	ae = HL_LSTM(config)
	ae.run(data_iterator, valid_data)
