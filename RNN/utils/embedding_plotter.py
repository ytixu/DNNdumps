import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from time import gmtime, strftime
from sklearn.decomposition import PCA as sklearnPCA
from tqdm import tqdm


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

def see_variational_length_embedding(encoder, decoder, data_iterator, validation_data, t, args):
	import metrics
	metrics.validate(validation_data, encoder, decoder, 10, metrics.ML_LSTM)

	# embedding = [[] for i in range(t)]
	# for x, y in data_iterator:
	# 	for i in range(t):
	# 		x_alter = np.zeros(x.shape)
	# 		x_alter[:,:i+1] = x[:,:i+1]
	# 		e = encoder.predict(x_alter)
	# 		if len(embedding[i]) == 0:
	# 			embedding[i] = e
	# 		else:
	# 			embedding[i] = np.concatenate((embedding[i], e), axis=0)
	# 	# break
	# l = len(embedding[0])
	# e_list = np.concatenate(embedding, axis=0)
	# embedding = np.array(embedding)
	# print embedding.shape
	# interpolate(embedding[t-1], encoder, decoder)
	# interpolate(embedding[t-1], encoder, decoder)
	# interpolate(embedding[t-1], encoder, decoder)
	# interpolate(embedding[t/2-1], encoder, decoder)
	# interpolate(embedding[t/2-1], encoder, decoder)
	# interpolate(embedding[t/2-1], encoder, decoder)
	# plot_convex_halls(e_list, l, t, args)
	# import metrics
	# metrics.plot_convex_hall(embedding, t, metrics.ML_LSTM)
	# go_up_hierarchy_vl(embedding, validation_data, encoder, decoder, t)
	# go_up_hierarchy_random_vl(embedding, validation_data, encoder, decoder, t)
	# go_up_hierarchy_random_test_bound_vl(embedding, validation_data, encoder, decoder, t)
	# ch = get_convex_hall(embedding[-1])
	# go_up_hierarchy_convexhall_vl(embedding, ch, validation_data, encoder, decoder, t/2-1)
	# go_up_hierarchy_dist(embedding, validation_data, encoder, decoder, t)

def see_hierarchical_embedding(encoder, decoder, data_iterator, validation_data, args):
	import metrics, metric_baselines
	metric_baselines.compare(encoder, decoder, 20, metrics.Prior_LSTM)
	# metrics.validate(validation_data, encoder, decoder, 20, metrics.Prior_LSTM)
	# # get_hierarchical_distances(encoder, data_iterator)
	# # # plot_centers(decoder)
	# # plot_communities(decoder)
	# # go_up_hierarchy_enc(encoder, decoder, data_iterator, 6)
	# embedding = []
	# h = 0
	# for x, y in data_iterator:
	# 	e = encoder.predict(x)
	# 	if len(embedding) == 0:
	# 		embedding = e
	# 		h = e.shape[1]
	# 	else:
	# 		embedding = np.concatenate((embedding, e), axis=0)
	# 	# break
	# l = embedding.shape[0]
	# e_list = np.concatenate(embedding, axis=0)
	# print embedding.shape, e_list.shape
	# # print'saving.....'
	# # np.save('../data.embedding_%s.npy'%strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()), embedding)

	# # check_min_max(e_list, decoder, h, args)
	# # subembedding(embedding[:,h-1], args)
	# # embedding_stats(embedding, args)
	# # plot_fa(e_list)
	# # interpolate(embedding[:,h-1], encoder, decoder)
	# # interpolate(embedding[:,h/2-1], encoder, decoder)
	# # plot_convex_halls(e_list, l, h, args)
	# import metrics
	# # metrics.plot_convex_hall(embedding, h, metrics.H_LSTM)
	# # metrics.go_up_hierarchy_sim(embedding, validation_data, encoder, decoder, h, metrics.H_LSTM, cut=h/2-1)
	# # metrics.gen_long_sequence(embedding, validation_data, encoder, decoder, h, metrics.H_LSTM)
	# # metrics.gen_long_sequence(embedding, validation_data, encoder, decoder, h, metrics.H_LSTM)
	# metrics.distance_stats(embedding, encoder, decoder, h, metrics.H_LSTM)
	# # idx = np.random.choice(len(validation_data), 100, replace=False)
	# # eval_go_up_hierarchy(embedding, validation_data[idx], encoder, decoder, h/2-1)
	# # go_up_hierarchy(embedding, validation_data, encoder, decoder, h)
	# # ch = get_convex_hall(embedding[:-1])
	# # ch = embedding[:5, -1]
	# # go_up_hierarchy_convexhall(embedding, ch, validation_data, encoder, decoder, h/2-1)
	# # go_up_hierarchy_random(embedding, validation_data, encoder, decoder, h)
	# # go_up_hierarchy_random_test_bound(embedding, validation_data, encoder, decoder, h)
	# # go_up_hierarchy_sim(embedding, validation_data, encoder, decoder, h)
	# # go_up_hierarchy_dist(embedding, validation_data, encoder, decoder, h)
	# # check_concept_space(embedding, h)
	# # k_mean_clusters(e_list, decoder)

def get_hierarchical_distances(encoder, data_iterator):
	i = 0
	for x, y in data_iterator:
		idx = np.random.choice(x.shape[0], 10)
		e = encoder.predict(x[idx])
		np.save('../out/embedding_list_%d.npy'%i, e)
		np.save('../out/embedding_orig_list_%d.npy'%i, x[idx])
		print i, e.shape
		i += 1

def check_concept_space(embedding, levels):
	from scipy.spatial import Delaunay
	from tqdm import tqdm

	polygones = []
	for i in tqdm(range(levels)):
		hull = Delaunay(embedding[:,i])
		p = embedding[:,i+1:]
		if len(p) > 0:
			n = np.random.choice(len(p), 10)
			for j in n:
				assert(hull.find_simplex(p[j])==-1)
		p = embedding[:,:i]
		if len(p) > 0:
			n = np.random.choice(len(p), 10)
			for j in n:
				assert(hull.find_simplex(p[j])==-1)

def plot_centers(decoder):
	import image
	# centers = np.load('../data/src/model_anal/centers.npy')
	# print centers.shape
	# p_poses = decoder.predict(centers)
	# for i in range(len(p_poses)/5+1):
	# 	image.plot_poses(p_poses[i*5:(i+1)*5])


	# centers = np.load('../data/src/model_anal/centers_orig.npy')
	# centers = 
	# # p_poses = decoder.predict(centers)
	# for i in range(len(centers)/5+1):
	# 	image.plot_poses(centers[i*5:(i+1)*5])


	centers = np.load('../data/src/model_anal/centers_orig_0.npy')
	for i in range(len(centers)/5):
		image.plot_poses(centers[i*5:(i+1)*5])

def k_mean_clusters(embedding, decoder, n=10):
	import image
	from sklearn.cluster import KMeans
	n_random = np.random.choice(len(embedding), 2000)
	kmeans = KMeans(n_clusters=n, random_state=0).fit(embedding[n_random])
	poses = decoder.predict(kmeans.cluster_centers_)
	for i in range(len(poses)/5):
		image.plot_poses(poses[i*5:(i+1)*5])

def plot_communities(decoder):
	import image
	import glob
	for std in [2]:
		# for c_files in glob.glob('../data/src/model_anal/H_LSTM_l100_t10/communities/community_*_orig_%dstd.npy'%(std)):
		# 	args = '_orig_'+str(std)+'_'+ c_files.split('_')[-3]+'_'
		# 	poses = np.load(c_files)
		# 	for i in range(len(poses)/5):
		# 		image.plot_poses(poses[i*5:(i+1)*5], args=args)

		for c_files in glob.glob('../data/src/model_anal/H_LSTM_l100_t10/communities/community_*_emb_vector_%dstd.npy'%(std)):
			args = '_gen_'+str(std)+'_'+ c_files.split('_')[-4]+'_'
			emb = np.load(c_files)
			poses = decoder.predict(emb)
			for i in range(len(poses)/5):
				image.plot_poses(poses[i*5:(i+1)*5], args=args)




def plot_convex_halls(embedding, sample_n, levels, args):
	transformed = pca_reduce(embedding)
	for i in range(levels):
		choices = transformed[i::levels]
		hull = ConvexHull(choices)
		for simplex in hull.simplices:
			plt.plot(choices[simplex, 0], choices[simplex, 1], c='#'+('%s'%('{0:01X}'.format(i+2)))*6)
		# plt.scatter(choices[:,0], choices[:,1], c='#'+('%s'%('{0:01X}'.format(i+2)))*6)
	
	for c in ['red', 'green', 'blue']:
		n = np.random.randint(0, sample_n-50)
		for i, s in enumerate(['x', '+', 'd']):
			random_sequence = (transformed[i*(levels-1)/2::levels])[n:n+50:10]
			print random_sequence.shape
			plt.scatter(random_sequence[:,0], random_sequence[:,1], c=c, marker=s)
	# plt.scatter(lda_transformed[y==2][0], lda_transformed[y==2][1], label='Class 2', c='blue')
	# plt.scatter(lda_transformed[y==3][0], lda_transformed[y==3][1], label='Class 3', c='lightgreen')

	# Display legend and show plot
	# plt.legend(loc=3)
	plt.savefig('../out/plot_convex_halls_'+ '-'.join(map(str, args)) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close()
	# plt.show()

def get_convex_hall(embedding):
	hull = ConvexHull(embedding)
	ch = None 
	for simplex in hull.simplices:
		return embedding[simplex]


def go_up_hierarchy_convexhall_vl(embedding, convex_hall, validation_data, encoder, decoder, cut=1):
	import image
	for _ in tqdm(range(10)):
		n = np.random.randint(len(validation_data))
		pose = np.copy(validation_data[n])
		pose[cut+1:] = 0
		enc = encoder.predict(np.array([pose]))
		z_ref = enc[0]
		zs = [z_ref]
		weights = [np.linalg.norm(convex_hall[j]-z_ref) for j in range(len(convex_hall))]
		w_i = np.argsort(weights)[:30]
		new_e = np.sum([convex_hall[d]/weights[d] for d in w_i], axis=0)/np.sum([1.0/weights[d] for d in w_i])
		zs.append(new_e)
		p_poses = decoder.predict(np.array(zs))
		image.plot_poses([pose, validation_data[n]], p_poses, title='Pattern matching (convex hall) (prediction in bold)')

def go_up_hierarchy_random_vl(embedding, validation_data, encoder, decoder, h):
	import metrics
	metrics.gen_random(embedding, validation_data, encoder, decoder, h, metrics.ML_LSTM)

def go_up_hierarchy_convexhall(embedding, convex_hall, validation_data, encoder, decoder, cut=1):
	import image
	for _ in tqdm(range(10)):
		n = np.random.randint(len(validation_data))
		enc = encoder.predict(validation_data[n:n+1])
		pose = np.copy(validation_data[n])
		pose[cut+1:] = 0
		z_ref = enc[0,cut]
		zs = [z_ref]
		weights = [np.linalg.norm(convex_hall[j]-z_ref) for j in range(len(convex_hall))]
		w_i = np.argsort(weights)[:30]
		new_e = np.sum([convex_hall[d]/weights[d] for d in w_i], axis=0)/np.sum([1.0/weights[d] for d in w_i])
		zs.append(new_e)
		p_poses = decoder.predict(np.array(zs))
		image.plot_poses([pose, validation_data[n]], p_poses, title='Pattern matching (convex hall) (prediction in bold)')

def go_up_hierarchy_random_test_bound_vl(embedding, validation_data, encoder, decoder, h):
	import metrics 
	metrics.plot_metrics(embedding, validation_data, encoder, decoder, h, metrics.ML_LSTM, 3)

def go_up_hierarchy_random_test_bound(embedding, validation_data, encoder, decoder, h):
	import metrics 
	metrics.plot_metrics(embedding, validation_data, encoder, decoder, h, metrics.H_LSTM)

def go_up_hierarchy_random(embedding, validation_data, encoder, decoder, h):
	import metrics 
	metrics.gen_random(embedding, validation_data, encoder, decoder, h, metrics.H_LSTM)

def go_up_hierarchy(embedding, validation_data, encoder, decoder, h, q=4):
	import image
	for _ in range(q):
		n = np.random.randint(len(validation_data))
		enc = encoder.predict(validation_data[n:n+1])
		pose = np.copy(validation_data[n])
		p_poses = decoder.predict(enc[0])
		image.plot_poses(np.array([pose]), p_poses, title='')
		
		for cut in range(h-1):
			pose[cut+1:] = 0
			z_ref = enc[0,cut]
			zs_sim = [z_ref, enc[0,-1]]
			zs_dist= [z_ref, enc[0,-1]]
			for i in range(1,h,2):
				e = embedding[:,i]
				weights = [np.linalg.norm(e[j]-z_ref) for j in range(len(e))]
				zs_sim.append(e[np.argmin(weights)])

				w_i = np.argsort(weights)[:30]
				new_e = np.sum([e[d]/weights[d] for d in w_i], axis=0)/np.sum([1.0/weights[d] for d in w_i])
				zs_dist.append(new_e)

			sim_poses = decoder.predict(np.array(zs_sim))
			dist_poses = decoder.predict(np.array(zs_dist))

			image.plot_poses(sim_poses[:2], sim_poses[2:], title='Pattern matching (closest)')
			image.plot_poses(dist_poses[:2], dist_poses[2:], title='Pattern matching (mean)')

def go_up_hierarchy_vl(embedding, validation_data, encoder, decoder, h, q=4):
	import image
	for _ in range(q):
		n = np.random.randint(len(validation_data))
		pose = np.reshape(np.repeat(validation_data[n], h, axis=0), (h, h, -1))
		for i in range(h):
			pose[i,i+1:] = 0
		enc = encoder.predict(pose)
		p_poses = decoder.predict(enc)
		image.plot_poses(pose[-1:], p_poses, title='')
		
		for cut in range(h-1):
			z_ref = enc[cut]
			zs_sim = [z_ref, enc[-1]]
			zs_dist= [z_ref, enc[-1]]
			for i in range(1,h,2):
				e = embedding[i]
				weights = [np.linalg.norm(e[j]-z_ref) for j in range(len(e))]
				zs_sim.append(e[np.argmin(weights)])

				w_i = np.argsort(weights)[:30]
				new_e = np.sum([e[d]/weights[d] for d in w_i], axis=0)/np.sum([1.0/weights[d] for d in w_i])
				zs_dist.append(new_e)

			sim_poses = decoder.predict(np.array(zs_sim))
			dist_poses = decoder.predict(np.array(zs_dist))

			image.plot_poses(sim_poses[:2], sim_poses[2:], title='Pattern matching (closest)')
			image.plot_poses(dist_poses[:2], dist_poses[2:], title='Pattern matching (mean)')


def go_up_hierarchy_sim(embedding, random_data, encoder, decoder, h, cut=1):
	import image
	n = np.random.randint(len(validation_data))
	enc = encoder.predict(validation_data[n:n+1])
	pose = np.copy(validation_data[n])
	pose[cut+1:] = 0
	z_ref = enc[0,cut]
	zs = [z_ref]
	for i in range(1,h,2):
		e = embedding[:,i]
		weights = [np.linalg.norm(e[j]-z_ref) for j in range(len(e))]
		zs.append(e[np.argmin(weights)])
	p_poses = decoder.predict(np.array(zs))
	image.plot_poses([pose, validation_data[n]], p_poses, title='Pattern matching (best) (prediction in bold)')

def go_up_hierarchy_dist(embedding, validation_data, encoder, decoder, h, cut=1):
	import image
	n = np.random.randint(len(validation_data))
	enc = encoder.predict(validation_data[n:n+1])
	pose = np.copy(validation_data[n])
	pose[cut+1:] = 0
	z_ref = enc[0,cut]
	zs = [z_ref]
	for i in range(1,h,2):
		e = embedding[:,i]
		weights = [np.linalg.norm(e[j]-z_ref) for j in range(len(e))]
		w_i = np.argsort(weights)[:30]
		new_e = np.sum([e[d]/weights[d] for d in w_i], axis=0)/np.sum([1.0/weights[d] for d in w_i])
		zs.append(new_e)
	p_poses = decoder.predict(np.array(zs))
	image.plot_poses([pose, validation_data[n]], p_poses, title='Pattern matching (mean) (prediction in bold)')

def go_up_hierarchy_enc(encoder, decoder, data_iterator, h):
	import image
	for x, y in data_iterator:
		n = np.random.randint(len(x))
		x_ref = np.copy(x[n:n+1])
		zs = []
		for i in range(h,0,-1):
			print x_ref.shape
			x_ref[:,i:] = 0
			x_enc = encoder.predict(x_ref)
			zs.append(x_enc[0,-1])
		p_poses = decoder.predict(np.array(zs))
		image.plot_poses(p_poses, x[n:n+1])


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
	image.plot_poses(t_poses, p_poses, title="Interpolation")


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
	plt.savefig('../out/check_min_max_'+ '-'.join(map(str, args)) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close()

def plot_fa(embedding):
	import networkx as nx
	from fa2 import ForceAtlas2
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

	plt.savefig('../out/embedding_stats_'+ '-'.join(map(str, args)) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close()
	# plt.show()

def subembedding(embedding, args):
	transformed = pca_reduce(embedding)
	plt.scatter(transformed[:,0], transformed[:,1])
	plt.savefig('../out/subembedding_'+ '-'.join(map(str, args)) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
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
