import numpy as np
import matplotlib.pyplot as plt
import random
# from mpl_toolkits.mplot3d import Axes3D
from time import gmtime, strftime


# def plot_time_series(x, title):
# 	plt.scatter(x, range(len(x)))
# 	plt.suptitle(title)
# 	plt.show()

def get_color(n):
	c = [None]*n
	for i in range(n):
		r = lambda: random.randint(0,255)
		c[i] = '#%02X%02X%02X' % (r(),r(),r())
	return c

def add_line(plt_canvas, coords, color, size):
	plt_canvas.plot(coords[:,1], coords[:,2], color=color, linewidth=size)

def add_point(plt_canvas, coords, color, size):
	l = len(coords)/2
	plt_canvas.scatter(x=coords[:,0], y=coords[:,2], c=color, s=size)
	# # plt_canvas.axis('off')
	# plt_canvas.axes.get_xaxis().set_ticks([])
	# plt_canvas.axes.get_yaxis().set_ticks([])
	# plt_canvas.set_xlim(min_lim, max_lim)
	# plt_canvas.set_ylim(min_lim, max_lim)


def plot_batch_3D(batch_true, batch_predict, title='Poses (prediction in blue)'):
	size = len(batch_true[0])
	f, axarr = plt.subplots(4, size, sharex=True, sharey=True)
	n = len(batch_true[0][0])/3
	for i, x in enumerate(batch_true[0]):
		x_reshaped = x.reshape((n, 3))
		add_point(axarr[0, i], x_reshaped[:,:2], 'r', 10)
		add_point(axarr[2, i], x_reshaped[:,1:], 'r', 10)
		x_reshaped = batch_predict[0][i].reshape((n, 3))
		add_point(axarr[1, i], x_reshaped[:,:2], 'b', 10)
		add_point(axarr[3, i], x_reshaped[:,1:], 'b', 10)

	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)

	f.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png')
	plt.close(f)

# def plot_batch_2D(batch_true, batch_predict, title='Poses (prediction in blue)'):
# 	size = len(batch_true[0])
# 	f, axarr = plt.subplots(2, size, sharex=True, sharey=True)
# 	for i, p in enumerate(batch_true[0]):
# 		new_p = np.reshape(p, (-1,2))
# 		add_point(axarr[0, i], new_p, 'r', 10)
# 		new_p = np.reshape(batch_predict[0][i], (-1,2))
# 		add_point(axarr[1, i], new_p, 'b', 10)

# 	f.subplots_adjust(hspace=0.1)
# 	plt.suptitle(title)

# 	f.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png')
# 	plt.close(f)


def plot_batch_2D(batch_complete, batch_true, batch_predict, best_opt, p_best_opt, title='Poses (Green, Cyan: predicted best option)'):
	size = len(batch_true[0])
	f, axarr = plt.subplots(len(batch_predict[0])+1, size, sharex=True, sharey=True)
	for i, p in enumerate(batch_true[0]):
		new_p = np.reshape(p, (-1,2))
		pose = np.reshape(batch_complete[0][i], (-1, 2))
		add_point(axarr[0, i], new_p, 'r', 10)
		add_point(axarr[0, i], pose, 'y', 10)

		for j in range(len(batch_predict[0])):
			new_p = np.reshape(batch_predict[0][j][i], (-1,2))
			c = 'b'
			if j == best_opt[0]:
				if best_opt[0] == p_best_opt[0]:
					c = 'g'
				else:
					c = 'm'
			elif j == p_best_opt[0]:
				c = 'c'
			add_point(axarr[j+1, i], new_p, c, 10)
			add_point(axarr[j+1, i], pose, 'y', 10)

	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)

	f.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png')
	plt.close(f)

def plot_batch_1D(batch_true, batch_predict, title='Poses (prediction in blue)'):
	size = len(batch_true[0])
	n = len(batch_true[0][0])
	f, axarr = plt.subplots(2, size, sharex=True, sharey=True)
	new_x = np.zeros((n, 2))
	new_x[:,0] = range(n)
	for i, x in enumerate(batch_true[0]):
		new_x[:,1] = x
		add_point(axarr[0, i], new_x, 'r', 10)
		new_x[:,1] = batch_predict[0][i]
		add_point(axarr[1, i], new_x, 'b', 10)

	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)

	f.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png')
	plt.close(f)

def plot_graph(batch_x, batch_true, batch_predict, best_opt=[], p_best_opt=[], title='Poses (Green, Cyan: predicted best option)'):
	t = np.arange(len(batch_true[0]))
	for i, p in enumerate(batch_true):
		plt.plot(t+batch_x[i].flatten(), p.flatten(), c='r')
	for i, p in enumerate(batch_predict):
		for j in range(len(p)):
			c = 'b'
			if len(best_opt) > 0:
				if j == best_opt[i]:
					if best_opt[i] == p_best_opt[i]:
						c = 'g'
					else:
						c = 'm'
				elif j == p_best_opt[i]:
					c = 'c'

			plt.plot(t+batch_x[i].flatten(), p[j].flatten(), c=c)
	plt.suptitle(title)
	plt.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png')
	plt.close()

def plot_options(ref, opts, title='Options'):
	colors = get_color(len(opts[0])+1)
	size = len(ref[0])
	n = len(ref[0][0])
	new_x = np.zeros((n*size, 2))
	new_x[:,0] = np.concatenate([np.arange(size)*1.0/(n+1)+i for i in range(n)])
	print new_x[:,0]
	new_x[:,1] = ref[0].T.flatten()
	add_point(plt, new_x, colors[0], 10)
	for j, opt in enumerate(opts[0]):
		new_x[:,1] = opt.T.flatten()
		add_point(plt, new_x, colors[j+1], 10)

	plt.suptitle(title)
	plt.show()
	# plt.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png')
	# plt.close(f)

# M_POSE_LINES = {'r':[1, 0],
# 				'g':[0, 2, 3, 4, 5],
# 				'b':[0, 6, 7, 8, 9],
# 				'm':[0, 10, 11, 12],
# 				'k':[0, 13, 14, 15]}

M_POSE_LINES = {'r':[0,1,2,3],
				'g':[0,4,5,6],
				'b':[0,7,8,9,10],
				'm':[8,11,12,13],
				'k':[9,14,15,16]}


def plot_hierarchies(batch_true, batch_predict, title='Reference is the first line'):
	timesteps = len(batch_true[0])
	hierarchies = len(batch_predict[0])/timesteps
	n = 5
	skip_t = timesteps/n
	skip_h = hierarchies/n
	f, axarr = plt.subplots(n+1, n, sharex=True, sharey=True)
	for i in range(n):
		new_x = np.reshape(batch_true[0][i*skip_t], (-1, 3))
		for c, l in M_POSE_LINES.iteritems():
			add_line(axarr[0, i], new_x[l], c, 3)

		for j in range(n):
			new_x = np.reshape(batch_predict[0][j*skip_h*hierarchies+i*skip_t], (-1, 3))
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[j+1, i], new_x[l], c, 3)

	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
	# plt.show()
	f.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png')
	plt.close(f)

def plot_options_hierarchies(batch_true, batch_predict, title='Reference is the first line'):
	print batch_predict.shape
	batch, hierarchies, options, _ = batch_predict.shape
	_, timesteps, _ = batch_true.shape
	batch_predict = np.reshape(batch_predict, (batch, hierarchies, options, timesteps,-1))
	n = 5
	skip_t = timesteps/n
	timestamp = strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime())
	for h in range(0, hierarchies, 2):
		f, axarr = plt.subplots(options+1, n, sharex=True, sharey=True)
		for i in range(n):
			new_x = np.reshape(batch_true[0][i*skip_t], (-1, 3))
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[0, i], new_x[l], c, 3)

			for j in range(options):
				new_x = np.reshape(batch_predict[0,h,j,i*skip_t], (-1, 3))
				for c, l in M_POSE_LINES.iteritems():
					add_line(axarr[j+1, i], new_x[l], c, 3)

		f.subplots_adjust(hspace=0.1)
		plt.suptitle(title)
		# plt.show()
		f.savefig('../out/' + title.lower().replace(' ', '_') + timestamp + '--%d'%(h) + '.png')
		plt.close(f)


def plot_poses(batch, batch2=[], title='Poses', args='', image_dir=None):
	timesteps = len(batch[0])
	n = len(batch)
	n_total = len(batch) + len(batch2)
	skip_n = timesteps
	skip_t = timesteps/skip_n
	f, axarr = plt.subplots(n_total, skip_n, sharex=True, sharey=True)

	for i in range(n):
		new_x = np.reshape(batch[i], (timesteps, -1, 3))
		for j in range(skip_n):
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[i, j], (new_x[j*skip_t])[l], c, 1)

	for i in range(len(batch2)):
		new_x = np.reshape(batch2[i], (timesteps, -1, 3))
		for j in range(skip_n):
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[i+n, j], (new_x[j*skip_t])[l], c, 2)


	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
	# plt.show()
	if image_dir == None:
		image_dir = '../out/'
	f.savefig(image_dir + title.lower().replace(' ', '_')+args+ strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png', dpi=192)
	plt.close(f)


def plot_poses_euler(batch, batch2=[], title='Poses', args='', image_dir=None):
	timesteps = len(batch[0])
	n = len(batch)
	n_total = len(batch) + len(batch2)
	skip_n = timesteps
	skip_t = timesteps/skip_n
	f, axarr = plt.subplots(n_total, skip_n, sharex=True, sharey=True)

	pose_template = np.zeros((timesteps, batch.shape[-1]+3))

	for i in range(n):
		pose_template[:,3:] = batch[i]
		new_x = np.reshape(pose_template, (timesteps, -1, 3))
		for j in range(skip_n):
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[i, j], (new_x[j*skip_t])[l], c, 1)

	for i in range(len(batch2)):
		pose_template[:,3:] = batch2[i]
		new_x = np.reshape(pose_template, (timesteps, -1, 3))
		for j in range(skip_n):
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[i+n, j], (new_x[j*skip_t])[l], c, 2)

	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
	# plt.show()
	if image_dir == None:
		image_dir = '../out/'
	f.savefig(image_dir + title.lower().replace(' ', '_')+args+ strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png', dpi=192)
	plt.close(f)



################### EULER ANGLES

def _some_variables():
	"""
	We define some variables that are useful to run the kinematic tree

	Args
	None
	Returns
	parent: 32-long vector with parent-child relationships in the kinematic tree
	offset: 96-long vector with bone lenghts
	rotInd: 32-long list with indices into angles
	expmapInd: 32-long list with indices into expmap angles
	"""

	parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9,10, 1,12,13,14,15,13,
					   17,18,19,20,21,20,23,13,25,26,27,28,29,28,31])-1

	offset = np.array([0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000])
	offset = offset.reshape(-1,3)

	rotInd = [[5, 6, 4],
		[8, 9, 7],
		[11, 12, 10],
		[14, 15, 13],
		[17, 18, 16],
		[],
		[20, 21, 19],
		[23, 24, 22],
		[26, 27, 25],
		[29, 30, 28],
		[],
		[32, 33, 31],
		[35, 36, 34],
		[38, 39, 37],
		[41, 42, 40],
		[],
		[44, 45, 43],
		[47, 48, 46],
		[50, 51, 49],
		[53, 54, 52],
		[56, 57, 55],
		[],
		[59, 60, 58],
		[],
		[62, 63, 61],
		[65, 66, 64],
		[68, 69, 67],
		[71, 72, 70],
		[74, 75, 73],
		[],
		[77, 78, 76],
		[]]

	expmapInd = np.split(np.arange(4,100)-1,32)

	return parent, offset, rotInd, expmapInd

def euler2mat(ai, aj, ak, axes='sxyz'):
	# https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py#L164
	"""Return rotation matrix from Euler angles and axis sequence.
	Parameters
	----------
	ai : float
	First rotation angle (according to `axes`).
	aj : float
	Second rotation angle (according to `axes`).
	ak : float
	Third rotation angle (according to `axes`).
	axes : str, optional
	Axis specification; one of 24 axis sequences as string or encoded
	tuple - e.g. ``sxyz`` (the default).
	Returns
	-------
	mat : array-like shape (3, 3) or (4, 4)
	Rotation matrix or affine.
	Examples
	--------
	>>> R = euler2mat(1, 2, 3, 'syxz')
	>>> np.allclose(np.sum(R[0]), -1.34786452)
	True
	>>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
	>>> np.allclose(np.sum(R[0]), -0.383436184)
	True
	"""
	try:
		firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
	except (AttributeError, KeyError):
		_TUPLE2AXES[axes]  # validation
		firstaxis, parity, repetition, frame = axes

	i = firstaxis
	j = _NEXT_AXIS[i+parity]
	k = _NEXT_AXIS[i-parity+1]

	if frame:
		ai, ak = ak, ai
	if parity:
		ai, aj, ak = -ai, -aj, -ak

	si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
	ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
	cc, cs = ci*ck, ci*sk
	sc, ss = si*ck, si*sk

	M = np.eye(3)
	if repetition:
		M[i, i] = cj
		M[i, j] = sj*si
		M[i, k] = sj*ci
		M[j, i] = sj*sk
		M[j, j] = -cj*ss+cc
		M[j, k] = -cj*cs-sc
		M[k, i] = -sj*ck
		M[k, j] = cj*sc+cs
		M[k, k] = cj*cc-ss
	else:
		M[i, i] = cj*ck
		M[i, j] = sj*sc-cs
		M[i, k] = sj*cc+ss
		M[j, i] = cj*sk
		M[j, j] = sj*ss+cc
		M[j, k] = sj*cs-sc
		M[k, i] = -sj
		M[k, j] = cj*si
		M[k, k] = cj*ci
	return M


def euler2rotmap(euler):
	return euler2mat(euler[0], euler[1], euler[2])

def rotmat2euler( R ):
	"""
	Converts a rotation matrix to Euler angles
	Matlab port to python for evaluation purposes
	https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1
	Args
	R: a 3x3 rotation matrix
	Returns
	eul: a 3x1 Euler angle representation of R
	"""
	if R[0,2] == 1 or R[0,2] == -1:
		# special case
		E3   = 0 # set arbitrarily
		dlta = np.arctan2( R[0,1], R[0,2] );

		if R[0,2] == -1:
			E2 = np.pi/2;
			E1 = E3 + dlta;
		else:
			E2 = -np.pi/2;
			E1 = -E3 + dlta;

	else:
		E2 = -np.arcsin( R[0,2] )
		E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
		E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

	eul = np.array([E1, E2, E3]);
	return eul

def revert_coordinate_space(channels, R0, T0):
	"""
	Bring a series of poses to a canonical form so they are facing the camera when they start.
	Adapted from
	https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

	Args
	channels: n-by-99 matrix of poses
	R0: 3x3 rotation for the first frame
	T0: 1x3 position for the first frame
	Returns
	channels_rec: The passed poses, but the first has T0 and R0, and the
	      rest of the sequence is modified accordingly.
	"""
	n, d = channels.shape

	channels_rec = copy.copy(channels)
	R_prev = R0
	T_prev = T0
	rootRotInd = np.arange(3,6)

	# Loop through the passed posses
	for ii in range(n):
		R_diff = data_utils.euler2rotmap( channels[ii, rootRotInd] )
		R = R_diff.dot( R_prev )

		channels_rec[ii, rootRotInd] = data_utils.rotmat2euler(R)
		T = T_prev + ((R_prev.T).dot(np.reshape(channels[ii,:3],[3,1]))).reshape(-1)
		channels_rec[ii,:3] = T
		T_prev = T
		R_prev = R

	return channels_rec

def fkl( angles, parent, offset, rotInd, expmapInd ):
	"""
	Convert joint angles and bone lenghts into the 3d points of a person.
	Based on expmap2xyz.m, available at
	https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

	Args
		angles: 99-long vector with 3d position and 3d joint angles in expmap format
		parent: 32-long vector with parent-child relationships in the kinematic tree
		offset: 96-long vector with bone lenghts
		rotInd: 32-long list with indices into angles
		expmapInd: 32-long list with indices into expmap angles
	Returns
		xyz: 32x3 3d points that represent a person in 3d space
	"""

	assert len(angles) == 99

	# Structure that indicates parents for each joint
	njoints   = 32
	xyzStruct = [dict() for x in range(njoints)]

	for i in np.arange( njoints ):

		if not rotInd[i] : # If the list is empty
			xangle, yangle, zangle = 0, 0, 0
		else:
			xangle = angles[ rotInd[i][0]-1 ]
			yangle = angles[ rotInd[i][1]-1 ]
			zangle = angles[ rotInd[i][2]-1 ]

		r = angles[ expmapInd[i] ]

		thisRotation = data_utils.euler2rotmap(r)
		thisPosition = np.array([xangle, yangle, zangle])

		if parent[i] == -1: # Root node
			xyzStruct[i]['rotation'] = thisRotation
			xyzStruct[i]['xyz']      = np.reshape(offset[i,:], (1,3)) + thisPosition
		else:
			xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot( xyzStruct[ parent[i] ]['rotation'] ) + xyzStruct[ parent[i] ]['xyz']
			xyzStruct[i]['rotation'] = thisRotation.dot( xyzStruct[ parent[i] ]['rotation'] )

	xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
	xyz = np.array( xyz ).squeeze()
	xyz = xyz[:,[0,2,1]]
	# xyz = xyz[:,[2,0,1]]


	return np.reshape( xyz, [-1] )


def plot_fk_from_euler(euler_angles, title='poses', image_dir='../new_out'):
	parent, offset, rotInd, expmapInd = _some_variables()
	euler_angles = revert_coordinate_space( euler_angles, np.eye(3), np.zeros(3) )
	xyz = fkl( euler_angles, parent, offset, rotInd, expmapInd )
	plot_poses_euler(xyz, title=title, image_dir=image_dir)