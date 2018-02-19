import collections 
import numpy as np
import embedding_plotter

MAX_MEAN_DIST = np.arange(10)/50.0

def visualize(y_data, y_encoded, y_decoded, opt):
	print collections.Counter(opt.tolist())
	idx_ref = np.random.choice(y_data.shape[0], 1)
	y_ref = y_data[idx_ref].flatten()
	
	idx = {k:[] for k in MAX_MEAN_DIST}

	for i, y in enumerate(y_data):
		diff = np.mean(np.absolute(y.flatten()-y_ref))
		
		for thr in MAX_MEAN_DIST:
			if diff < thr:
				idx[thr].append(i)

	print idx
	for k, indices in idx.iteritems():
		print [k, 
			np.mean(np.absolute(np.mean(y_encoded[indices], axis=0) - y_encoded[idx_ref])), 
			np.mean(np.absolute(y_encoded[indices] - y_encoded[idx_ref])),
			y_encoded[indices].shape]
		# embedding_plotter.plot_points(y_encoded, indices)
