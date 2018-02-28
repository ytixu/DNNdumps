import json
import numpy as np
import glob
import os.path
from scipy.spatial import distance
import matplotlib.pyplot as plt
import random

from sklearn.decomposition import PCA as sklearnPCA

end_count = 20
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

ref = np.array([[-0.,-0.06400316,-0.13125549,-0.18743867,0.,0.8265,0.5884453,0.5607307,0.,-0.14909998,-0.36195703,-0.63159509],
	[-0.,-0.06408317,-0.08681219,-0.06308334,0.,0.8265,0.57959338,0.54030711,0.,-0.15148401,-0.38082593,-0.64645837],
	[-0.,-0.06149184,-0.03949452,0.0530897,0.,0.8265,0.59517447,0.57323712,0.,-0.13990808,-0.37280945,-0.61457776],
	[-0.,-0.06903787,-0.01493435,0.13286532,0.,0.8265,0.66051291,0.716621,0.,-0.14534814,-0.40437964,-0.60659436],
	[-0.,-0.06770515,-0.08009978,0.00116798,0.,0.8265,0.72553529,0.89342841,0.,-0.16127197,-0.42768701,-0.60475244],
	[-0.,-0.07142286,-0.12216105,-0.10452317,0.,0.8265,0.71311848,0.86693926,0.,-0.15838123,-0.42345142,-0.62883472],
	[-0.,-0.06196463,-0.09405238,-0.08625121,0.,0.8265,0.60023923,0.61528172,0.,-0.1477688,-0.39800403,-0.65694214]])

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
	#
	X = pca_reduce()
	# X = np.concatenate([e[1] for e in sequence])[:,2]
	idx = 0
	for i, seq in enumerate(sequence):
		y,_ = seq
		x = X[idx:idx+len(y)].flatten()
		for j in range(7):
			plt.plot(x+j*0.5, y[:,j], c=cl[i])
		idx += len(y)
	plt.show()

	# x = np.concatenate([np.arange(ref_length)/8.0+i for i in range(7)])
	# print x
	# for i, match in enumerate(matching_sequences):
	# 	if matching_sequences_dist[i] > 0.30:
	# 		plt.scatter(x, match/2/np.pi, c='#f0f0f0')
	# for i, match in enumerate(matching_sequences):
	# 	if matching_sequences_dist[i] > 0.30:
	# 		continue
	# 	elif matching_sequences_dist[i] > 0.25:
	# 		plt.scatter(x, match/2/np.pi, c='#c0c0c0')
	# for i, match in enumerate(matching_sequences):
	# 	if matching_sequences_dist[i] > 0.25:
	# 		continue
	# 	elif matching_sequences_dist[i] > 0.20:
	# 		plt.scatter(x, match/2/np.pi, c='#909090')
	# for i, match in enumerate(matching_sequences):
	# 	if matching_sequences_dist[i] > 0.20:
	# 		continue
	# 	plt.scatter(x, match/2/np.pi)
	# plt.show()

def dist_ref(seq):
	return distance.euclidean(ref, np.array(seq).flatten())

filenames = []
for datafile in glob.glob('raw_data/*.json'):
	filenames.append(int(datafile.split('raw_data/kp-')[1].split('-')[0]))

filenames = sorted(list(set(filenames)))

for file_idx in filenames:
	if file_idx < 1519249593:
		continue
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

			subseq = subseq + [format(data['poses_seq'][str(i)]) for i in range(count)]
			pose_count += count

		idx += 1
		if idx > 100:
			break

	if len(sequence) < 1:
		sequence = subseq[start:end_count]
		init_pose = sequence[0]['h']
	else:
		sequence = sequence + subseq[start:end_count]

	sequences[colors] = subseq[start:end_count]
	colors += 1

print 'sequences', colors
# print len(matching_sequences)
# sequence = normalize()
sequence = _reformat(sequence)
plot()
