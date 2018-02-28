import json
import numpy as np
import glob
import os.path

pose_count = 0
seq_count = 0

for datafile in glob.glob('raw_data/*'):
	print datafile
	data = json.load(open(datafile, 'r'))
	r_data = np.array([data['poses_seq'][str(i)]['r'] for i in range(data['count'])])
	h_data = np.array([np.array(data['poses_seq'][str(i)]['h']).flatten() for i in range(data['count'])])

	seq_count += 1
	pose_count += data['count']

	filename = os.path.basename(datafile).split('.')[0]
	np.save('../seq_robot/'+filename+'.npy', r_data)
	np.save('../seq_human/'+filename+'.npy', h_data)

print seq_count, pose_count