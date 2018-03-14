import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from time import gmtime, strftime
import networkx as nx
from sklearn.decomposition import PCA as sklearnPCA
from fa2 import ForceAtlas2


def see_embedding(encoder, data_iterator, args, concat=False):
	embedding = np.array([])
	for x, y in data_iterator:
		if concat:
			x = np.concatenate((x,y), axis=2)
		
		e = encoder.predict(x)

		if len(embedding) == 0:
			embedding = e
		else:
			embedding = np.concatenate((embedding, e), axis=0)

	plot(embedding, args)

def see_hierarchical_embedding(encoder, decoder, data_iterator, args):
	embedding = []
	h = 0
	for x, y in data_iterator:
		e = encoder.predict(x)
		if len(embedding) == 0:
			embedding = e
			h = e.shape[1]
		else:
			embedding = np.concatenate((embedding, e), axis=0)

	l = embedding.shape[0]
	# print'saving.....'
	# np.save('../data.embedding_%s.npy'%strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()), embedding)

	# check_min_max(e_list, decoder, h, args)
	# subembedding(embedding[:,h-1], args)
	# embedding_stats(embedding, args)
	# plot_fa((embedding[:,h-1])[:50], args)
	# interpolate(embedding[:,h-1], encoder, decoder)
	go_up_hierarchy(embedding, decoder, h)

	# transformed = pca_reduce(e_list)
	# for i in range(h):
	# 	choices = transformed[i::h]
	# 	hull = ConvexHull(choices)
	# 	for simplex in hull.simplices:
	# 		plt.plot(choices[simplex, 0], choices[simplex, 1], c='#'+('%s'%('{0:01X}'.format(i+2)))*6)
	# 	# plt.scatter(choices[:,0], choices[:,1], c='#'+('%s'%('{0:01X}'.format(i+2)))*6)
	
	# for c in ['red', 'green', 'blue']:
	# 	n = np.random.randint(0, l-50)
	# 	for i, s in enumerate(['x', '+', 'd']):
	# 		random_sequence = (transformed[i*(h-1)/2::h])[n:n+50:10]
	# 		print random_sequence.shape
	# 		plt.scatter(random_sequence[:,0], random_sequence[:,1], c=c, marker=s)
	# # plt.scatter(lda_transformed[y==2][0], lda_transformed[y==2][1], label='Class 2', c='blue')
	# # plt.scatter(lda_transformed[y==3][0], lda_transformed[y==3][1], label='Class 3', c='lightgreen')

	# # Display legend and show plot
	# # plt.legend(loc=3)
	# plt.savefig('../out/embedding_'+ '-'.join(map(str, args)) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	# plt.close()
	# # plt.show()

def go_up_hierarchy(embedding, decoder, h):
	import image
	n = np.random.randint(len(embedding[:,0]))
	z_ref = (embedding[:,0])[n]
	zs = [z_ref]
	for i in range(2,h,2):
		e = embedding[:,i]
		weights = [np.linalg.norm(e[j]-z_ref) for j in range(len(e))]
		zs.append(e[np.argmin(weights)])
	p_poses = decoder.predict(np.array(zs))
	image.plot_poses(p_poses[:1], p_poses[1:])


def interpolate(embedding, encoder, decoder, l=8):
	import image
	n = np.random.choice(len(embedding), 2)
	dist = (embedding[n[1]] - embedding[n[0]])/l
	zs = [embedding[n[0]]]
	for i in range(l):
		zs.append(zs[0]+i*dist)
	zs.append(embedding[n[1]])
	zs = np.array(zs)
	t_poses = decoder.predict(zs)
	# image.plot_poses(t_poses)

	x1 = np.concatenate([t_poses[:,0], t_poses[-1,1:]], axis=0)[::2]
	x2 = np.concatenate([t_poses[0,:], t_poses[1:,-1]], axis=0)[::2]
	x3 = np.array([t_poses[i,i] for i in range(10)])

	print x1.shape, x2.shape, x3.shape
	x_poses = np.array([x1,x2,x3])
	zs_pred = encoder.predict(x_poses)
	p_poses = decoder.predict(zs_pred[:,-1])
	image.plot_poses(t_poses, p_poses)


def k_mean(embedding, decoder, n=8):
	from sklearn.cluster import KMeans
	import image
	kmeans = KMeans(n_clusters=n, random_state=0).fit(embedding)
	zs_pred = kmeans.cluster_centers_
	zs = [None]*len(zs_pred)
	for i,z in enumerate(zs_pred):	
		zs[i] = embedding[np.argmin([np.linalg.norm(embedding[j]-z) for j in range(len(embedding))])]
	zs = np.array(zs)
	t_poses = decoder.predict(zs)
	print t_poses.shape
	image.plot_poses(t_poses)


def check_min_max(embedding, decoder, h, args):
	#### PLOTING THE POSES
	# n = np.random.randint(len(embedding))
	# weights = [[i, np.linalg.norm(embedding[i]-embedding[n])] for i in range(len(embedding)) if i != n]
	# weights = sorted(weights, key=lambda x: x[-1])
	# p_poses = decoder.predict((embedding[[w for w,_ in weights[::10]]])[:h])
	# t_poses = decoder.predict(embedding[n:n+1])
	# p_poses = np.reshape(p_poses, (1, h**2, -1))
	
	# import image
	# image.plot_hierarchies(t_poses, p_poses, title='Similarity (red is reference)')

	#### HISTOGRAMs
	# ns = np.random.choice(len(embedding), 5, replace=False)
	# f, axarr = plt.subplots(len(ns), 1, sharex=True, sharey=True)
	# for i,n in enumerate(ns):
	# 	weights = [np.linalg.norm(embedding[j]-embedding[n]) for j in range(len(embedding)) if j != n]
	# 	axarr[i].hist(weights, bins=100)

	hist = []
	bin_edges = None
	idx = np.random.choice(len(embedding), 20, replace=False)
	for i,a in enumerate(embedding[idx]):
		print i
		weights = [np.linalg.norm(a-b) for j, b in enumerate(embedding) if j > i]
		if len(hist) == 0:
			hist, bin_edges = np.histogram(weights, bins=100)
		else:
			hist += np.histogram(weights, bins=bin_edges)[0]
	
	plt.bar(bin_edges[:-1], hist, width = 1)
	plt.xlim(min(bin_edges), max(bin_edges))
	plt.savefig('../out/embedding_'+ '-'.join(map(str, args)) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close()

def plot_fa(embedding):
	fa2 = ForceAtlas2()
	G = nx.DiGraph()
	weights = [(i, j, np.linalg.norm(embedding[i]-embedding[j])) for i in range(len(embedding)) for j in range(i)]
	G.add_weighted_edges_from(weights)
	positions = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
	nx.draw_networkx(G, positions, cmap=plt.get_cmap('jet'), node_size=50, with_labels=False, width=1, arrows=False)
	plt.show()

def embedding_stats(embedding, args):
	# r = plt.hist(embedding, density=True)
	# print r
	# plt.savefig('../out/embedding_'+ '-'.join(map(str, args)) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	# plt.close()
	f, axarr = plt.subplots(embedding.shape[1], 1, sharex=True, sharey=True)

	x = np.arange(embedding.shape[-1])
	for i in range(embedding.shape[1]):
		y = np.mean(embedding[:,i], axis=0)
		err = np.std(embedding[:,i], axis=0)
		axarr[i].plot(x, y, c='#'+('%s'%('{0:01X}'.format(i+2)))*6)
		axarr[i].fill_between(x, y-err, y+err, facecolor='#'+('%s'%('{0:01X}'.format(i+2)))*6)

	plt.savefig('../out/embedding_'+ '-'.join(map(str, args)) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close()
	# plt.show()

def subembedding(embedding, args):
	transformed = pca_reduce(embedding)
	plt.scatter(transformed[:,0], transformed[:,1])
	plt.savefig('../out/embedding_'+ '-'.join(map(str, args)) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close()


def pca_reduce(embedding):
	pca = sklearnPCA(n_components=2) #2-dimensional PCA
	X_norm = (embedding - embedding.min())/(embedding.max() - embedding.min())
	transformed = pca.fit_transform(X_norm)
	return transformed	

def plot_points(embedding, indices):
	transformed = pca_reduce(embedding)
	plt.scatter(transformed[:,0], transformed[:,1], c='blue')
	plt.scatter(transformed[indices,0], transformed[indices,1], c='red')
	plt.show()


def plot(embedding, args):
	transformed = pca_reduce(embedding)
	plt.scatter(transformed[:,0], transformed[:,1], c='blue')
	
	for c in ['red', 'lightgreen', 'yellow']:
		n = np.random.randint(0, len(transformed)-10)
		random_sequence = transformed[n:n+10]	
		plt.scatter(random_sequence[:,0], random_sequence[:,1], c=c)
	# plt.scatter(lda_transformed[y==2][0], lda_transformed[y==2][1], label='Class 2', c='blue')
	# plt.scatter(lda_transformed[y==3][0], lda_transformed[y==3][1], label='Class 3', c='lightgreen')

	# Display legend and show plot
	# plt.legend(loc=3)
	plt.savefig('../out/embedding_'+ '-'.join(map(str, args)) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close()

	# plt.show()

if __name__ == '__main__':
	embedding = np.load('../../data/data.embedding_Tue-13-Mar-2018-19_36_37.npy')
	print 'loaded.....'
	e = np.concatenate(embedding[:100], axis=0)
	print e.shape
	plot_fa(e)
