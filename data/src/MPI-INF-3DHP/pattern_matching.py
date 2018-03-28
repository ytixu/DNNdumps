import numpy as np
import glob
from tqdm import tqdm
from time import gmtime, strftime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
import json 


load_path = '../../mpi/1st_camera/*.npy'
t = 6

M_POSE_LINES = {'r':[1, 0],
				'g':[0, 2, 3, 4, 5],
				'b':[0, 6, 7, 8, 9],
				'm':[0, 10, 11, 12],
				'k':[0, 13, 14, 15]}

def error(x, y):
	return np.mean(np.square(x - y))

def add_line(plt_canvas, coords, color, size):
	plt_canvas.plot(coords[:,0], 1-coords[:,1], color=color, linewidth=size)

def save_pose(pose, id):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	pose = np.reshape(pose, (-1,3))
	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)
	ax.set_zlim(-1, 1)
	xs = pose[:,0]
	ys = pose[:,2]
	zs = -pose[:,1]
	for c, l in M_POSE_LINES.iteritems():
		ax.plot(xs[l], ys[l], zs[l], color=c)

	plt.savefig('./out/pose_'+ str(id) + '.png') 
	plt.close()

def plot_poses(batch, title='Poses'):
	n = len(batch) 
	if n > 0:
		f, axarr = plt.subplots(n, t, sharex=True, sharey=True)

		for i in range(n):
			new_x = np.reshape(batch[i], (t, -1, 3))
			for j in range(t):
				for c, l in M_POSE_LINES.iteritems():
					if n > 1:
						add_line(axarr[i, j], new_x[j,l,:2], c, 3)
					else:
						add_line(axarr[j], new_x[j,l,:2], c, 3)


		f.subplots_adjust(hspace=0.1)
		plt.suptitle(title)
		plt.show()
		# f.savefig(strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
		# plt.close(f)


def linear_motion_stats(poses):
	n = len(poses)-t
	std = [0]*(n)
	for i in range(n):
		diff = poses[i:i+t-1]-poses[i+1:i+t]
		std[i] = np.std(diff)
	print np.mean(std), np.min(std)

def filter_linear_motion(poses, std=0.013):
	n = len(poses)-t
	f_poses = []
	for i in range(n):
		diff = poses[i:i+t-1]-poses[i+1:i+t]
		if np.std(diff) < std:
			f_poses.append(poses[i:i+t])
	plot_poses(np.array(f_poses))
	print len(f_poses)

def __expand_t(poses):
	n = len(poses)-t+1
	e_poses = np.zeros((n,t,poses.shape[-1]))
	for i in range(n):
		e_poses[i,:] = poses[i:i+t]
	return e_poses

def get_distance_mat(cut=-1):
	poses = []
	dist_mat = {}
	first = True
	for filename in tqdm(glob.glob(load_path)):
		p = np.load(filename)
		p = __expand_t(p)
		if len(poses) == 0:
			poses = p
		else:
			poses = np.concatenate((poses, p), axis=0)
		n = len(p)
		for i in range(n):
			if first:
				first = False
				continue
			t_poses = poses[:-n+i,:cut]
			dist_mat[i] = distance_matrix(np.reshape(p[i:i+1,:cut], (1, -1)), np.reshape(t_poses, (t_poses.shape[0],-1)))[0].tolist()
		# break
	return dist_mat, poses

def get_graph_communities(dist_mat):
	import networkx as nx
	import community

	_mean = 0.5
	G = nx.Graph()
	for i, dist in dist_mat.iteritems():
		weights = [(i,j,_mean-d) for j,d in enumerate(dist) if d < _mean]
		G.add_weighted_edges_from(weights)

	print 'Number of components', nx.number_connected_components(G)

	part = community.best_partition(G)
	mod = community.modularity(part,G)
	# values = [part.get(node) for node in G.nodes()]
	# nx.draw_spring(G, cmap=plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)
	# plt.show()
	# plt.savefig('graph_%dstd.png'%std_n)
	# plt.close()
	return {node: part.get(node) for node in G.nodes()}

# saving pictures
# for filename in glob.glob(load_path):
# 	poses = np.load(filename)
# 	for i,p in enumerate(tqdm(poses)):
# 		save_pose(p, i)
# 	break

selected_patterns = [
	[86, 92], # walking 
	[96, 102],
	[577, 583],
	[942, 948], # lunge
	[972, 978],
	[1144, 1150], # push-up
	[1160, 1166],
	[1354, 1360], # exercises
	[1364, 1370], 
	[1432, 1438],
	[1562, 1568],
	[1744, 1750], # sitting
	[1799, 1805],
	[1824, 1830],
	[1862, 1868],
	[1886, 1892],
	[1932, 1938]
]

patterns = {}
matched = {}
for filename in glob.glob(load_path):
	poses = np.load(filename)
	if len(patterns) == 0:
		for i, p in enumerate(selected_patterns):
			patterns[i] = poses[p[0]:p[1]]
			matched[i] = []
		continue
	n = len(poses)-t+1
	for i in tqdm(range(n)):
		for j in patterns:
			diff = error(poses[i:i+t], patterns[j])
			if diff < 0.001:
				matched[j].append([poses[i:i+t]])

	for i in patterns:
		plot_poses([patterns[i]] + matched[i][:5])



# dm, poses = get_distance_mat()
# with open('get_distance_mat.txt', 'w') as outfile:
# 	json.dump({'distance-mat': dm, 'poses': poses.tolist()}, outfile)

# with open('get_distance_mat.txt', 'r') as infile:
# 	data = json.load(infile)
# 	dm, poses = data['distance-mat'], data['poses']
# 	comm = get_graph_communities(dm)
# 	with open('get_graph_communities.txt', 'w') as outfile:
# 		json.dump(comm, outfile)

# poses = None
# comm = None
# with open('get_distance_mat.txt', 'r') as infile:
# 	poses = np.array(json.load(infile)['poses'])
# with open('get_graph_communities.txt', 'r') as infile:
# 	comm = json.load(infile)

# for i in tqdm(set(comm.values())):
# 	nodes = [n for n,c in comm.iteritems() if c == i]
# 	for j in range(len(nodes)/5):
# 		idx = map(int, nodes[j*5:(j+1)*5])
# 		plot_poses(poses[idx], title='Community %d'%(i))

# dm, poses = get_distance_mat(3)
# with open('get_distance_mat_cut3.txt', 'w') as outfile:
# 	json.dump({'distance-mat': dm, 'poses': poses.tolist()}, outfile)