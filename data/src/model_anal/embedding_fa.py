import numpy as np
import csv
import matplotlib.pyplot as plt
# from time import gmtime, strftime
from scipy.spatial import ConvexHull
import networkx as nx
from fa2 import ForceAtlas2
import community
import glob
from tqdm import tqdm

# id for embedding : fileId_arrayId_hierarchyId
# id for original data: fileId_arrayId

DATA_FOLDER = './H_LSTM_l100_t10/data/'
SAVE_DIST_FOLDER = './H_LSTM_l100_t10/dist-npy/'

MEAN = 1.0767261003
VAR = 0.337042815032

def compute_distace_files():
	global MEAN, VAR
	mean = []
	std = []

	def compute_dist_mat(data_a, data_b, file_a, file_b):
		b,h,d = data_a.shape
		filename = SAVE_DIST_FOLDER+'distance_list_%s_%s.npy'%(file_a, file_b)
		n = b*h
		data_a = np.reshape(data_a, (n,d))
		data_b = np.reshape(data_b, (n,d))
		dist_mat = np.zeros((n, n))
		# with open(filename, 'wb') as csvfile:
		# 	spamwriter = csv.writer(csvfile)
		# 	spamwriter.writerow(['Source', 'Target', 'Weight'])
		# 	for i in range(b):
		# 		for j in range(h):
		# 			for k in range(b):
		# 				for l in range(h):
		# 					spamwriter.writerow(['%s_%d_%d'%(file_a,i,j), '%s_%d_%d'%(file_b,k,l), np.linalg.norm(data_a[i][j] - data_b[k][l])])			
		for i in range(n):
			for j in range(n):
				dist_mat[i,j] = np.linalg.norm(data_a[i] - data_b[j])
		np.save(filename, dist_mat)
		m, s = np.mean(dist_mat), np.std(dist_mat)
		mean.append(m)
		std.append(s)
		

	filenames = glob.glob(DATA_FOLDER+'embedding_list*.npy')
	for i, f in enumerate(tqdm(filenames)):
		data = np.load(f)
		idx = f.split('_')[-1].split('.')[0]
		compute_dist_mat(data, data, idx, idx)
		for j, ff in enumerate(tqdm(filenames)):
			if j > i:
				data_b = np.load(f)
				idx_b = ff.split('_')[-1].split('.')[0]
				compute_dist_mat(data, data_b, idx, idx_b)

	MEAN, VAR = np.mean(mean), np.mean(std)

def compute_graph(std_n = 3):
	G = nx.Graph()
	
	filenames = glob.glob(SAVE_DIST_FOLDER+'*.npy')
	for fi, f in enumerate(tqdm(filenames)):
		idxs = map(int, f.split('.')[-2].split('_')[-2:])
		data = np.load(f)
		weights = [(idxs[0]*10000+i, idxs[1]*10000+j, e) for i,d in enumerate(data) for j,e in enumerate(d) if e < (MEAN-std_n*VAR) and i!=j]
		G.add_weighted_edges_from(weights)
		# if fi > 2:
		# break

	print 'Number of components', nx.number_connected_components(G)
	# print nx.graph_number_of_cliques(G)
	nx.write_gml(G, 'graph_%dstd.gml'%std_n)

def get_fa(std_n = 3):
	G = nx.read_gml('graph_%dstd.gml'%std_n)
	fa2 = ForceAtlas2()
	positions = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
	nx.draw_networkx(G, positions, cmap=plt.get_cmap('jet'), node_size=50, with_labels=False, width=1, arrows=False)
	plt.show()

def __get_embedding_lists():
	nodes = {}
	for f in tqdm(glob.glob(DATA_FOLDER + 'embedding_list*.npy')):
		idx = int(f.split('_')[-1].split('.')[0])
		data = np.load(f)
		b,t,d = data.shape
		nodes[idx] = np.reshape(data, (b*t,d))
	return nodes

def __get_embedding_orig_lists():
	nodes = {}
	for f in tqdm(glob.glob(DATA_FOLDER + 'embedding_orig_list_*.npy')):
		idx = int(f.split('_')[-1].split('.')[0])
		data = np.load(f)
		nodes[idx] = data
	return nodes
	

def get_components(std_n = 3):
	nodes = __get_embedding_lists()
	G = nx.read_gml('graph_%dstd.gml'%std_n)
	centers = []
	for cpn in tqdm(nx.connected_components(G)):
		c_nodes = np.array([nodes[int(node)/10000][int(node)%10000] for node in cpn])
		centers.append([np.mean(c_nodes, axis=0), len(c_nodes)])

	centers = sorted(centers, key=lambda x: x[1], reverse=True)
	np.save('centers.npy', [c[0] for c in centers])

def get_orig_components(std_n = 3):
	nodes = __get_embedding_orig_lists()
	G = nx.read_gml('graph_%dstd.gml'%std_n)
	centers = []
	for cpn in tqdm(nx.connected_components(G)):
		c_nodes = np.array([nodes[int(node)/10000][(int(node)%10000)/10] for node in cpn])
		for i, node in enumerate(cpn):
			idx = (int(node)%10000)%10
			c_nodes[i,idx+1:] = 0

		centers.append([np.mean(c_nodes, axis=0), len(c_nodes)])

	centers = sorted(centers, key=lambda x: x[1], reverse=True)
	np.save('centers_orig.npy', [c[0] for c in centers])

def get_one_components(std_n = 3):
	nodes = __get_embedding_orig_lists()
	G = nx.read_gml('graph_%dstd.gml'%std_n)
	centers = []
	for c_idx, cpn in enumerate(tqdm(nx.connected_components(G))):
		c_nodes = np.array([nodes[int(node)/10000][(int(node)%10000)/10] for node in cpn])
		for i, node in enumerate(cpn):
			idx = (int(node)%10000)%10
			c_nodes[i,idx+1:] = 0
			
		np.save('centers_orig_%d.npy'%(c_idx), c_nodes)
		break


def find_communities(std_n = 3):
	G = nx.read_gml('graph_%dstd.gml'%std_n)
	# Find modularity
	part = community.best_partition(G)
	mod = community.modularity(part,G)
	# Plot, color nodes using community structure
	values = [part.get(node) for node in G.nodes()]
	# nx.draw_spring(G, cmap=plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)
	# plt.savefig('graph_%dstd.png'%std_n)
	# plt.close()

	nodes = __get_embedding_lists()
	values = set(values)
	for i in tqdm(values):
		com = [node for node in G.nodes() if part.get(node) == i]
		c_nodes = np.array([nodes[int(node)/10000][(int(node)%10000)] for node in com])
		np.save('community_%d_emb_vector_%dstd.npy'%(i, std_n), c_nodes)

	# nodes = __get_embedding_orig_lists()
	# for i in tqdm(values):
	# 	com = [node for node in G.nodes() if part.get(node) == i]
	# 	c_nodes = np.array([nodes[int(node)/10000][(int(node)%10000)/10] for node in com])
	# 	for j, node in enumerate(com):
	# 		idx = (int(node)%10000)%10
	# 		c_nodes[j,idx+1:] = 0
	# 	np.save('community_%d_orig_%dstd.npy'%(i, std_n), c_nodes)


# mean and var : 1.08535868536 0.31154654317
# compute_distace_files()
# compute_graph(1)
# compute_graph(2)
# get_fa(1)
# get_fa(2)
# get_one_components(1)
find_communities(2)
find_communities(1)

# mean = []
# std = []
# for f in tqdm(glob.glob('./*/embedding_orig_list_*.npy')):
# 	idx = int(f.split('_')[-1].split('.')[0])
# 	data = np.load(f)
# 	normalized = [np.linalg.norm(data[i,0]-data[i,1]) for i in range(len(data))]
# 	mean.append(np.mean(normalized))
# 	std.append(np.std(normalized))
# print np.mean(mean), np.mean(std)
# 0.0397236685401 0.0274501646596
