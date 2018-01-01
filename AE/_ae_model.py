import numpy as np
import time


class ae_model:

	def __init__(self, args):
		self.batch_size = args['batch_size']
		self.epochs = args['epochs']
		self.latent_dim = None
		self.output_dim = None
		self.input_dim = None
		self.input_intermediate_dim = None
		self.output_intermediate_dim = None
		self.epsilon_std = 1.0
		self.load_path = args['load_path']
		self.mode = args['mode']

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
