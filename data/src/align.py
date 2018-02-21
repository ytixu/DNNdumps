import json
import numpy as np
import glob
import os.path
from scipy.spatial import distance
import matplotlib.pyplot as plt
import random


end_count = 200
start = 0

sequence = []
sequences = {}
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

def normalize():
	for entry in sequence:
		entry['h'] = distance.euclidean(entry['h'], init_pose)
	seq = sorted(sequence, key=lambda x:x['h'])
	seq = [(np.array([e['r'] for e in seq if e['c'] == i]), np.array([e['h'] for e in seq if e['c'] == i])) for i in range(colors)]
	return seq

def plot():
	cl = get_color()
	# for i, seq in enumerate(sequence):
	# 	y, x = seq
	# 	for j in range(6):
	# 		plt.plot(x+j*1, y[:,j], c=cl[i])
	# plt.show()

	for i, seq in sequences.iteritems():
		x = np.arange(0,len(seq),5)*1.0/(end_count+0.2)
		y = np.array([s['r'] for k, s in enumerate(seq) if k%5 == 0])
		for j in range(6):
			plt.scatter(x+j, y[:,j], c=cl[i])
	plt.show()


filenames = []
for datafile in glob.glob('raw_data/*.json'):
	filenames.append(int(datafile.split('raw_data/kp-')[1].split('-')[0]))

filenames = sorted(list(set(filenames)))

for file_idx in filenames:
	# if file_idx < 1518363647:
		# continue
	pose_count = 0
	subseq = []
	idx = 0
	while pose_count < end_count:
		datafile = 'raw_data/kp-%d-%d.json'%(file_idx, idx)
		if os.path.isfile(datafile):
			print datafile
			data = json.load(open(datafile, 'r'))
			count = min(end_count, data['count'])
			subseq = subseq + [format(data['poses_seq'][str(i)]) for i in range(count)]
			pose_count += count
		idx += 1

	if len(sequence) < 1:
		sequence = subseq[start:end_count]
		init_pose = sequence[0]['h']
	else:
		sequence = sequence + subseq[start:end_count]

	sequences[colors] = subseq[start:end_count]
	colors += 1

print colors
sequence = normalize()
plot()
