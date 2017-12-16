import json
import numpy as np

def read(file_name, _H, _W, header=False):
	data = json.load(open(file_name, 'r'))
	mat = np.asfarray([v for k,v in data.iteritems() if k != u'header'])
	batch, l = mat.shape
	l = l/2
	X = np.zeros([batch,l,2,1])
	X[:,:,0,0] = (1-mat[:,:l])/_H
	X[:,:,1,0] = (1-mat[:,l:])/_W
	if header:
		return X, data[u'header']
	return X

if __name__ == '__main__':
	_W = 720
	_H = 480
	mat = read('data/open_poses.json', _H, _W)
	batch = np.random.choice(range(len(mat)), 16)
	import image
	image.plot_batch(mat[batch], 16)