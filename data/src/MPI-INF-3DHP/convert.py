import glob
import scipy.io
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

# max_3d = 0
# min_3d = 10000

# max_2d = 0
# min_2d = 10000

# max_3d = 7468.23890463 
# max_2d = 10427.0 
# min_3d = -4272.62907408 
# min_2d = 1-6249.2

# all_joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis', ...     %5       
#         'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', ... %11
#        'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', ... %17
#        'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', ...        %23   
#        'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe']; 

path = '/usr/local/data/yitian/mpi_inf_3dhp/*/*/annot.mat'
save_path = '../../mpi/'

# for filename in glob.glob(path):
# 	mat = scipy.io.loadmat(filename)
# 	for cam_id in range(mat['univ_annot3'].shape[0]):
# 		for pose in mat['univ_annot3'][cam_id][0]:
# 			max_3d = max(max_3d, np.max(pose))
# 			min_3d = min(min_3d, np.min(pose))
# 		for pose in mat['annot2'][cam_id][0]:
# 			max_2d = max(max_2d, np.max(pose))
# 			min_2d = min(min_2d, np.min(pose))



# print max_3d, max_2d, min_3d, min_2d
# n_3d = max_3d - min_3d
# n_2d = max_2d - min_2d
relevant_idx = [6, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25]
non_relevant_idx = list(set(range(28)) - set(relevant_idx))

r_arm = [14, 16] 
# [23, 14, 15, 16, 17]
non_r_arm = list(set(relevant_idx) - set(r_arm))

for f_id, filename in enumerate(glob.glob(path)):
	mat = scipy.io.loadmat(filename)
	for cam_id in range(mat['annot2'].shape[0]):

		# pose_3d = (mat['univ_annot3'][cam_id][0] - min_3d)/n_3d
		pose_2d = mat['annot2'][cam_id][0]
		min_2d = np.min(pose_2d, axis=1)
		max_2d = np.max(pose_2d, axis=1)
		for i in range(pose_2d.shape[0]):
			pose_2d[i] = (pose_2d[i] - min_2d[i])/(max_2d[i] - min_2d[i])
		# assert len(pose_3d) == len(pose_2d)

		# for pose in pose_3d:
		# 	fig = plt.figure()
		# 	ax = fig.add_subplot(111, projection='3d')
		# 	pose = np.reshape(pose, (-1,3))
		# 	ax.set_xlim(0, 1)
		# 	ax.set_ylim(0, 1)
		# 	ax.set_zlim(-1, 0)
		# 	xs = pose[:,0]
		# 	ys = pose[:,2]
		# 	zs = -pose[:,1]
		# 	ax.scatter(xs[relevant_idx], ys[relevant_idx], zs[relevant_idx], color='r')
		# 	ax.scatter(xs[non_relevant_idx], ys[non_relevant_idx], zs[non_relevant_idx], color='b')
		# 	plt.show()

		print cam_id
		# np.save(save_path+'3d/%d-%d.npy'%(f_id,cam_id), pose_3d[:,relevant_idx])
		# np.save(save_path+'2d/%d-%d.npy'%(f_id,cam_id), pose_2d[:,relevant_idx])
		pose_2d = np.reshape(pose_2d, (len(pose_2d), -1, 2))
		for i in range(pose_2d.shape[0]): 
			pose_2d[i] = pose_2d[i] - pose_2d[i][r_arm[0]]
		# arm_2d = pose_2d[:,r_arm]
		# for pose in arm_2d:
		# 	pose = np.reshape(pose, (-1,2))
		# 	xs = pose[:,0]
		# 	ys = -pose[:,1]
			# plt.xlim((0,1))
			# plt.ylim((-1,0))
			# plt.scatter(xs[r_arm][[0,-1]], ys[r_arm][[0,-1]], color='r')
			# plt.scatter(xs[r_arm][1:-1], ys[r_arm][1:-1], color='b')
			# plt.scatter(xs[non_r_arm], ys[non_r_arm], color='g')
			# plt.scatter(xs, ys, color='g')
			# plt.show()
		# print np.reshape(pose_2d[:,relevant_idx], (len(pose_2d), -1)).shape
		np.save(save_path+'arm_complete/%d-%d.npy'%(f_id,cam_id), np.reshape(pose_2d[:,non_r_arm], (len(pose_2d), -1)))
		np.save(save_path+'arm_partial/%d-%d.npy'%(f_id,cam_id), np.reshape(pose_2d[:,r_arm], (len(pose_2d), -1)))