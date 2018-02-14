import keras
import image

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

	def record(self, log_path, args):
		print 'logging ---------', log_path
		with open(log_path, 'a+') as log_file:
			header = ' - '.join(map(str, args)) + ' :: '
			log_file.write(header + ' '.join(map(str, self.losses)) + '\n')