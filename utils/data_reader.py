import json
import numpy as np

def rescale(x, w, h):
	batch, l = x.shape
	X = np.zeros(x.shape)
	l = l/2
	X[:,:l] = 1-(x[:,:l])/h
	X[:,l:] = 1-(x[:,l:])/w
	return X


def read(data_file, refe_file=None, _H=0, _W=0, header=False):
	data = json.load(open(data_file, 'r'))
	mat_data = []
	
	if refe_file:
		refe = json.load(open(refe_file, 'r'))
		mat_refe = []
	for k,v in data[u'poses'].iteritems():
		mat_data.append(v['pose'])
		if refe_file:
			mat_refe.append(refe[u'poses'][k])

	X = np.asfarray(mat_data)

	if refe_file:
		l = len(refe[u'header'])
		keys = np.array([refe[u'header'].index(i) for i in ['R_Hip', 'R_Shoulder','R_Elbow','R_Wrist','R_Hand']])
		keys = np.concatenate([keys, keys + l, keys + l*2]) 
		Y = np.asfarray(mat_refe)[:,keys]
	# X = rescale(X, _W, _H)
	# Y = rescale(Y, _W, _H)

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