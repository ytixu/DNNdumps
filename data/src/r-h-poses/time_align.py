import glob
import json
import numpy as np
import matplotlib.pyplot as plt

def format_pose(p):
	# return p
	return '-'.join(map(str, [k for i in p for k in i[1:]]))

h_poses = {}
file_count = 0
for filename in glob.glob('h_raw_data/*'):
	file_id = int(filename.split('.')[0].split('_')[-1])
	h_poses[file_id] = json.load(open(filename, 'r'))
	file_count = max(file_id, file_count)

h_poses = [format_pose(h_poses[i][str(j)][1]) for i in range(file_count + 1) for j in sorted(map(int, h_poses[i].keys()))]
r_poses = {}

def normalize(pose):
	return np.array(pose)# / 2 / np.pi %1

min_ = 1100
max_ = 0

for filename in glob.glob('raw_data/*'):
	file_ids = map(int, filename.split('.')[0].split('-')[-2:])
	# if file_ids[0] < 1519670558:
	if file_ids[0] < 1519249593:
	# if file_ids[0] < 1519510262:
		continue
	poses = json.load(open(filename, 'r'))
	last_idx = -1
	split_n = 0.0
	# last_pose = []
	for i in range(poses['count']):
		# p = normalize(poses['poses_seq'][str(i)]['r'])[:-1]
		# if len(last_pose) > 0:
		# 	diff = np.absolute(last_pose - p)
		# 	if max(diff) > 0.5:
		# 		if file_ids[0] in r_poses:
		# 			del r_poses[file_ids[0]]
		# 			break
		# last_pose = p
		pose = format_pose(poses['poses_seq'][str(i)]['h'])
		# print np.argmin([np.absolute(pose - p) for p in h_poses])
		idx = []
		found_idx = 0
		while True:
			try:
				idx.append(h_poses[found_idx:].index(pose))
				found_idx = idx[-1]+1
			except:
				break
		if len(idx) == 1:
			if idx[0] > 1100 or idx < 101:
				continue
			if last_idx > idx[0]:
				split_n += 0.001
			key = file_ids[1]+split_n
			if file_ids[0] not in r_poses:
				r_poses[file_ids[0]] = {key:[]}
			elif key not in r_poses[file_ids[0]]:
				r_poses[file_ids[0]][key] = []
			last_idx = idx[0]
			r_poses[file_ids[0]][key].append((idx[0], normalize(poses['poses_seq'][str(i)]['r'])))

# print r_poses
print len(r_poses)
n = 6
f, axarr = plt.subplots(n*2, 1, sharex=True, sharey=True)
for name,ps in r_poses.iteritems():
	print name
	for _,p in ps.iteritems():
		x = [t for t,_ in p]
		y = np.array([j for _,j in p])
		for i in range(n):
			axarr[i*2].plot(x, np.sin(y[:,i]))
			axarr[i*2+1].plot(x, np.cos(y[:,i]))

plt.show()
plt.close(f)
