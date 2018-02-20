import json
import numpy as np
import glob
import os.path
from scipy.spatial import distance
import matplotlib.pyplot as plt

pose_count = 20
start = 0

sequence = []
colors = 0
init_pose = None

def format(entry):
	entry['c'] = colors
	entry['h'] = np.array(entry['h'])[:,-1]
	return entry

def normalize():
	for entry in sequence:
		entry['h'] = distance.euclidean(entry['h'], init_pose)
	seq = sorted(sequence, key=lambda x:x['h'])
	seq = [(np.array([e['r'] for e in seq if e['c'] == i]), [e['h'] for e in seq if e['c'] == i]) for i in range(colors)]
	return seq

def plot():
	for i in range(7):
		for y, x in sequence:
			plt.plot(x, y[:,i])

		plt.show()



for datafile in glob.glob('raw_data/*-0.json'):
	print datafile
	data = json.load(open(datafile, 'r'))
	if data['count'] < pose_count:
		continue
	if len(sequence) < 1:
		sequence = [format(data['poses_seq'][str(i)]) for i in range(start, pose_count)]
		init_pose = sequence[0]['h']
	else:
		sequence = sequence + [format(data['poses_seq'][str(i)]) for i in range(start, pose_count)]

	colors += 1

print colors
sequence = normalize()
plot()
