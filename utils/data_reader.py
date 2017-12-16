import json
import numpy as np

def rescale(x, w, h):
	batch, l = x.shape
	X = np.zeros(x.shape)
	l = l/2
	X[:,:l] = 1-(x[:,:l])/h
	X[:,l:] = 1-(x[:,l:])/w
	return X


def read(data_file, refe_file, _H, _W, header=False):
	data = json.load(open(data_file, 'r'))
	refe = json.load(open(refe_file, 'r'))
	mat_data = []
	mat_refe = []
	for k,v in data.iteritems():
		if k != u'header':
			mat_data.append(data[k])
			mat_refe.append(refe[k])

	mat_data = np.asfarray(mat_data)
	mat_refe = np.asfarray(mat_refe)
	X = rescale(mat_data, _W, _H)
	Y = rescale(mat_refe, _W, _H)

	if header:
		return X, Y, data[u'header']
	return X, Y

if __name__ == '__main__':
	_W = 720
	_H = 480
	mat = read('data/open_poses.json', _H, _W)
	batch = np.random.choice(range(len(mat)), 16)
	import image
	image.plot_batch(mat[batch], 16)