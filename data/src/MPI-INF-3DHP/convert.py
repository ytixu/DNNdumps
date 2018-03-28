import glob
import scipy.io
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D


def rot_y(x, axis):
	rot_mat_y = np.zeros((3,3))
	rot_mat_y[1,1] = 1
	c = np.sqrt(axis[2]**2 + axis[0]**2)
	cos = axis[0]/c
	sin = axis[2]/c
	rot_mat_y[0] = np.array([cos, 0, sin])
	rot_mat_y[2] = np.array([-sin, 0, cos])
	axis = np.dot(rot_mat_y, axis)

	return np.dot(rot_mat_y, x.T).T

def rot_x(x, axis):
	rot_mat_y = np.zeros((3,3))
	rot_mat_y[1,1] = 1
	c = np.sqrt(axis[2]**2 + axis[0]**2)
	cos = axis[0]/c
	sin = axis[2]/c
	rot_mat_y[0] = np.array([cos, 0, sin])
	rot_mat_y[2] = np.array([-sin, 0, cos])
	axis = np.dot(rot_mat_y, axis)

	rot_mat_x = np.zeros((3,3))
	rot_mat_x[0,0] = 1
	c = np.sqrt(axis[1]**2 + axis[0]**2)
	cos = axis[0]/c
	sin = axis[1]/c
	rot_mat_x[1:,1:] = np.array([[cos, -sin], [sin, cos]])

	return np.dot(rot_mat_x, np.dot(rot_mat_y, x.T)).T



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

# neck, head, l_shoulder, l_elbow, l_wrist, l_hand, r_shoulder, r_elbow, r_wrist, r_hand, l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle
lines = [[1, 0, 2, 3, 4, 5],
		[0, 6, 7, 8, 9],
		[0, 10, 11, 12],
		[0, 13, 14, 15]]

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
relevant_idx = [5, 6, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25]
non_relevant_idx = list(set(range(28)) - set(relevant_idx))
MIN_DIFF = 0.0397236685401 + 2*0.0274501646596
r_arm = [14, 16] 
# [23, 14, 15, 16, 17]
non_r_arm = list(set(relevant_idx) - set(r_arm))
prev_pose = []

for f_id, filename in enumerate(glob.glob(path)):
	mat = scipy.io.loadmat(filename)
	for cam_id in range(mat['annot2'].shape[0]):
		for cam_view_id in range(mat['univ_annot3'][cam_id].shape[0]):
			pose_3d = mat['univ_annot3'][cam_id][cam_view_id]
			pose_3d = np.reshape(pose_3d, (len(pose_3d), -1, 3))[:,relevant_idx]
			valid_idx = []
			for i in range(len(pose_3d)):
				# recenter to the neck 
				pelvis = (pose_3d[i][10] + pose_3d[i][13])/2
				pose_3d[i] = (pose_3d[i] - pelvis)/np.linalg.norm(pose_3d[i][2]-pose_3d[i][6])/5
				# rotate
				pose_3d[i] = rot_x(pose_3d[i], pose_3d[i,10])
				# pose_3d[i] = rot_y(pose_3d[i], pose_3d[i,0])


				if len(prev_pose) == 0:
					prev_pose = pose_3d[i]
				else:
					if np.linalg.norm(pose_3d[i]-prev_pose) > MIN_DIFF:
						prev_pose = pose_3d[i]
						valid_idx.append(i)

			print cam_id, '%d/%d' %(len(valid_idx), len(pose_3d))
			pose_3d = pose_3d[valid_idx]

			# pose_2d = mat['annot2'][cam_id][0]
			# min_2d = np.min(pose_2d, axis=1)
			# max_2d = np.max(pose_2d, axis=1)
			# for i in range(pose_2d.shape[0]):
			# 	pose_2d[i] = (pose_2d[i] - min_2d[i])/(max_2d[i] - min_2d[i])
			# # assert len(pose_3d) == len(pose_2d)

			# for pose in pose_3d:
			# 	fig = plt.figure()
			# 	ax = fig.add_subplot(111, projection='3d')
			# 	pose = np.reshape(pose, (-1,3))
			# 	ax.set_xlim(-1, 1)
			# 	ax.set_ylim(-1, 1)
			# 	ax.set_zlim(-1, 1)
			# 	xs = pose[:,0]
			# 	ys = pose[:,2]
			# 	zs = -pose[:,1]
			# 	for l in lines:
			# 		ax.plot(xs[l], ys[l], zs[l], color='b')
			# 	# ax.scatter(xs[relevant_idx], ys[relevant_idx], zs[relevant_idx], color='r')
			# 	# ax.scatter(xs[-1:], ys[-1:], zs[-1:], color='b')
			# 	ax.plot([0, xs[0]], [0, ys[0]], [0, zs[0]], color='r')
			# 	ax.plot(xs[[10,13]], ys[[10,13]], zs[[10,13]], color='r')
			# 	plt.show()

			assert np.max(pose_3d) < 1
			assert np.min(pose_3d) > -1
			np.save(save_path+'1st_camera/%d-%d.npy'%(f_id,cam_id), np.reshape(pose_3d, (len(pose_3d), -1)))
		break
		# # np.save(save_path+'2d/%d-%d.npy'%(f_id,cam_id), pose_2d[:,relevant_idx])
		# pose_2d = np.reshape(pose_2d, (len(pose_2d), -1, 2))
		# for i in range(pose_2d.shape[0]): 
		# 	pose_2d[i] = pose_2d[i] - pose_2d[i][r_arm[0]]
		# # arm_2d = pose_2d[:,r_arm]
		# # for pose in arm_2d:
		# # 	pose = np.reshape(pose, (-1,2))
		# # 	xs = pose[:,0]
		# # 	ys = -pose[:,1]
		# 	# plt.xlim((0,1))
		# 	# plt.ylim((-1,0))
		# 	# plt.scatter(xs[r_arm][[0,-1]], ys[r_arm][[0,-1]], color='r')
		# 	# plt.scatter(xs[r_arm][1:-1], ys[r_arm][1:-1], color='b')
		# 	# plt.scatter(xs[non_r_arm], ys[non_r_arm], color='g')
		# 	# plt.scatter(xs, ys, color='g')
		# 	# plt.show()
		# # print np.reshape(pose_2d[:,relevant_idx], (len(pose_2d), -1)).shape
		# np.save(save_path+'arm_complete/%d-%d.npy'%(f_id,cam_id), np.reshape(pose_2d[:,non_r_arm], (len(pose_2d), -1)))
		# np.save(save_path+'arm_partial/%d-%d.npy'%(f_id,cam_id), np.reshape(pose_2d[:,r_arm], (len(pose_2d), -1)))