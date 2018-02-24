import glob
import json
import numpy as np

def format_pose(p):
	# return p
	return '-'.join(map(str, p))

h_poses = {}
file_count = 0
for filename in glob.glob('h_raw_data/*'):
	file_id = int(filename.split('.')[0].split('_')[-1])
	h_poses[file_id] = json.load(open(filename, 'r'))
	file_count = max(file_id, file_count)

h_poses = [format_pose(h_poses[i][str(j)][0]) for i in range(file_count + 1) for j in sorted(map(int, h_poses[i].keys()))]
print h_poses
r_poses = {}

for filename in glob.glob('raw_data/*'):
	file_ids = map(int, filename.split('.')[0].split('-')[-2:])
	poses = json.load(open(filename, 'r'))
	for _, p in poses['poses_seq'].iteritems():
		pose = format_pose(p['h'])
		# print np.argmin([np.absolute(pose - p) for p in h_poses])
		print pose
		# idx = []
		# while True:
		# 	try:
		# 		idx.append(h_poses.index(pose))
		# 	except:
		# 		break
		# print idx
