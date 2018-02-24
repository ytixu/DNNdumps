import glob
import json

h_poses = {}
file_count = 0
for filename in glob.glob('h_raw_data/*')
	file_id = int(filename.split('.')[0].split('_')[-1])
	h_poses[file_id] = json.load(open(filename, 'r'))
	file_count = max(file_id, file_count)

h_poses = [h_poses[i][j] for i in range(file_count + 1) for j in sorted(h_poses[i].keys())]

