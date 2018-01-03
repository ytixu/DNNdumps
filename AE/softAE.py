import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics, objectives

from _ae_model import ae_model
from DNNdumps.utils import arg_parse, image

class SoftAE(ae_model, object):

	def __init__(self, args):
		self.autoencoder = None
		self.encoder = None
		self.decoder = None
		super(AE, self).__init__(args)

	def set_input_dim(self, x, y):
		self.input_dim = len(x[0])
		self.output_dim = len(y[0])
		self.latent_dim = (self.input_dim+self.output_dim)/2
		self.input_intermediate_dim = (self.input_dim+self.latent_dim)/2
		self.output_intermediate_dim = (self.output_dim+self.latent_dim)/2