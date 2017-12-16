import numpy as np
import time


class gan_model:

	def __init__(self, args):
		self.batch_size = args['batch_size']
		self.epoch = args['epoch']
		self.latent_size = None
		self.input_shape = None
		self.X = None
		self.Z = None

	def set_data(self, x, y):
		self.X = x
		self.Z = y
		h, w, d = self.X[0].shape
		self.input_shape = (h, w, d, )
		h, w, d = self.Z[0].shape
		self.latent_shape = (h, w, d, )
		# self.norm_max = np.amax(np.amax(np.abs(self.X)))

	def start_timer(self):
		self.start_time = time.time()
	
	def elapsed(self,sec):
		if sec < 60:
			return str(sec) + " sec"
		elif sec < (60 * 60):
			return str(sec / 60) + " min"
		else:
			return str(sec / (60 * 60)) + " hr"

	def elapsed_time(self):
		print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )
