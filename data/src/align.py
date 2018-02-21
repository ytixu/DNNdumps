import json
import numpy as np
import glob
import os.path
from scipy.spatial import distance
import matplotlib.pyplot as plt
import random

from sklearn.decomposition import PCA as sklearnPCA

end_count = 300
start = 0

sequence = []
sequences = {}
matching_sequences = []
matching_sequences_dist = []
colors = 0
init_pose = None

def get_color():
	c = [None]*colors
	for i in range(colors):
		r = lambda: random.randint(0,255)
		c[i] = '#%02X%02X%02X' % (r(),r(),r())
	return c

def format(entry):
	entry['c'] = colors
	entry['h'] = np.array(entry['h'])[:,-1]
	return entry

def pca_reduce():
	pca = sklearnPCA(n_components=1)
	X = np.concatenate([e[1] for e in sequence])
	X_norm = (X - X.min())/(X.max() - X.min())
	return pca.fit_transform(X_norm)

def _reformat(seq):
	return [(np.array([e['r'] for e in seq if e['c'] == i]), np.array([e['h'] for e in seq if e['c'] == i])) for i in range(colors)]

def normalize():
	for entry in sequence:
		np.append(entry['h'], (distance.euclidean(entry['h'], init_pose)))
	seq = sorted(sequence, key=lambda x:x['h'][-1])
	return _reformat(seq)

ref = np.array([[-0.,-0.07765717,-0.35011166,-0.56932296,0.,0.8265,0.67871046,0.67477705,0.,-0.19268457,-0.25632532,-0.42875513],
	[-0.,-0.07900089,-0.35394524,-0.63903786,0.,0.8265,0.68305478,0.65486365,0.,-0.19246484,-0.25182092,-0.36776599],
	[-0.,-0.07476571,-0.3091818,-0.55153422,0.,0.8265,0.65089581,0.6348223,0.,-0.19378894,-0.26056458,-0.44473157],
	[-0.,-0.07321952,-0.26960053,-0.41750074,0.,0.8265,0.63462285,0.63496442,0.,-0.1946167,-0.2612124,-0.51932092],
	[-0.,-0.07423346,-0.27115487,-0.30292141,0.,0.8265,0.62468486,0.63116413,0.,-0.19530432,-0.26849707,-0.55359692]])

ref_length = len(ref)
ref = ref.flatten()

def plot():
	cl = get_color()
	# for i, seq in enumerate(sequence):
	# 	y, x = seq
	# 	for j in range(6):
	# 		plt.plot(x+j*1, y[:,j], c=cl[i])
	# plt.show()

	# for i, seq in sequences.iteritems():
	# 	x = np.arange(0,len(seq),5)*1.0/(end_count+0.2)
	# 	y = np.array([s['r'] for k, s in enumerate(seq) if k%5 == 0])
	# 	for j in range(6):
	# 		plt.scatter(x+j, y[:,j], c=cl[i])
	# plt.show()

	# X = pca_reduce()
	# # X = np.concatenate([e[1] for e in sequence])[:,2]
	# idx = 0
	# for i, seq in enumerate(sequence):
	# 	y,_ = seq
	# 	x = X[idx:idx+len(y)].flatten()
	# 	for j in range(7):
	# 		plt.scatter(x+j*0.5, y[:,j], c=cl[i])
	# 	idx += len(y)
	# plt.show()

	x = np.concatenate([np.arange(ref_length)/8.0+i for i in range(7)])
	print x
	for i, match in enumerate(matching_sequences):
		if matching_sequences_dist[i] > 0.35:
			plt.scatter(x, match/2/np.pi, c='#f0f0f0')
	for i, match in enumerate(matching_sequences):
		if matching_sequences_dist[i] > 0.35:
			continue
		elif matching_sequences_dist[i] > 0.3:
			plt.scatter(x, match/2/np.pi, c='#c0c0c0')
	for i, match in enumerate(matching_sequences):
		if matching_sequences_dist[i] > 0.3:
			continue
		elif matching_sequences_dist[i] > 0.27:
			plt.scatter(x, match/2/np.pi, c='#909090')
	for i, match in enumerate(matching_sequences):
		if matching_sequences_dist[i] > 0.27:
			continue
		plt.scatter(x, match/2/np.pi)
	plt.show()

def dist_ref(seq):
	return distance.euclidean(ref, np.array(seq).flatten())

filenames = []
for datafile in glob.glob('raw_data/*.json'):
	filenames.append(int(datafile.split('raw_data/kp-')[1].split('-')[0]))

filenames = sorted(list(set(filenames)))

for file_idx in filenames:
	# if file_idx < 1518363647:
	# 	continue
	pose_count = 0
	subseq = []
	idx = 0
	while pose_count < end_count:
		datafile = 'raw_data/kp-%d-%d.json'%(file_idx, idx)
		if os.path.isfile(datafile):
			print datafile
			data = json.load(open(datafile, 'r'))
			count = min(end_count, data['count'])
			if count >= ref_length:
				for l in range(count-ref_length+1):
					temp_seq = [data['poses_seq'][str(i)]['h'] for i in range(l, l+ref_length)]
					dist = dist_ref(temp_seq)
					if dist < 0.4:
						matching_sequences.append(np.array([data['poses_seq'][str(i)]['r'] for i in range(l, l+ref_length)]).T.flatten())
						matching_sequences_dist.append(dist)

			# subseq = subseq + [format(data['poses_seq'][str(i)]) for i in range(count)]
			pose_count += count

		idx += 1
		if idx > 100:
			break

	# if len(sequence) < 1:
	# 	sequence = subseq[start:end_count]
	# 	init_pose = sequence[0]['h']
	# else:
	# 	sequence = sequence + subseq[start:end_count]

	# sequences[colors] = subseq[start:end_count]
	colors += 1

print colors, len(matching_sequences)
# sequence = normalize()
# sequence = _reformat(sequence)
plot()
