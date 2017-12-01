import json
import numpy as np

_W = 720
_H = 480


def read(file_name):
	data = json.load(open(file_name, 'r'))
	mat = np.array([v for k,v in data.iteritems() if k is not 'header'])
	n = mat.shape[1]/2
	mat[:,:n] = mat[:,:n] / _H
	mat[:,n:] = mat[:,n:] / _W
	return mat