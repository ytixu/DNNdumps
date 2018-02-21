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

ref = np.array([[-0.,-0.03560034,-0.06574796,0.20380635,0.,0.8265,0.60281187,0.66822247,0.,-0.16868311,-0.30241724,-0.45005017],[-0.,-0.05100711,-0.09400163,0.05651277,0.,0.8265,0.67989624,0.93307932,0.,-0.17062439,-0.34221436,-0.46242981],[-0.,-0.06744191,-0.11358131,-0.05843297,0.,0.8265,0.66993254,0.93584393,0.,-0.17070239,-0.34890369,-0.4917937,],[-0.,-0.068549,-0.14301926,-0.1544953,0.,0.8265,0.63294165,0.8690079,0.,-0.18192029,-0.32567432,-0.50546936],[-0.,-0.05271277,-0.11189134,-0.10901235,0.,0.8265,0.60190224,0.60749255,0.,-0.1807168,-0.2474812,-0.56240491]])
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
