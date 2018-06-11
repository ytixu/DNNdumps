import numpy as np
from tqdm import tqdm
import metrics
import matplotlib.pyplot as plt

def e_dist(ea, eb):
	# return np.linalg.norm(ea - eb)
	return np.sum(np.abs(ea - eb))

def eval_sim_bw_levels(model, validation_data):
	h = model.timesteps
	# mean, var = np.zeros(len(model.hierarchies)), np.zeros(len(model.hierarchies))
	x = np.arange(model.timesteps) + 1
	encodings = metrics.__get_latent_reps(model.encoder, validation_data, model.MODEL_CODE)

	for c, cut in enumerate(tqdm(model.hierarchies)):
		mean_z, var_z = np.zeros(h), np.zeros(h)
		n = validation_data.shape[0]
		mean, var = np.zeros(h), np.zeros(h)
		# x1, x2 = np.zeros(validation_data[0].shape), np.zeros(validation_data[0].shape)

		# decodings_cut = metrics.__get_decoded_reps(model.decoder, encodings[:, cut], model.MODEL_CODE)
		for i in range(model.timesteps):
			diff = [e_dist(encodings[j, cut], encodings[j,i]) for j in range(n)]
			mean_z[i] = np.mean(diff)
			var_z[i] = np.std(diff)

			a = np.random.choice(n, 1000, replace=False)
			b = np.random.choice(n, 1000, replace=False)
			diff = [e_dist(encodings[j,cut], encodings[k,i]) for j in a for k in b]
			mean[i] = np.mean(diff)
			var[i] = np.std(diff)

		# a = np.random.choice(n, 1000, replace=False)
		# b = np.random.choice(n, 1000, replace=False)
		# diff = [np.linalg.norm(encodings[j,cut] - encodings[k,cut]) for j in a for k in b]
		# mean[c] = np.mean(diff)
		# var[c] = np.std(diff)

			# decodings = metrics.__get_decoded_reps(model.decoder, encodings[:, i], model.MODEL_CODE)
			# diff_x = np.zeros(n)
			# for j in range(n):
			# # 	x1[:cut] = validation_data[j, :cut]
			# # 	x1[cut:] = 0
			# # 	x2[:i] = validation_data[j, :i]
			# # 	x2[i:] = 0
			# 	x1 = decodings_cut[j]
			# 	x2 = decodings[j]
			# 	diff_x[j] = metrics.__pose_seq_error(x1, x2)

			# plt.scatter(diff_x, diff)
			# plt.xlabel('pose difference')
			# plt.ylabel('latent difference')
			# plt.title('diffence X vs Z (length %d, %d)'%(cut, i))
			# plt.show()

		p = plt.plot(x, mean_z, label=cut+1)
		plt.fill_between(x, mean_z-var_z, mean_z+var_z, alpha=0.3, color=p[-1].get_color())
		plt.plot(x, mean, linestyle='dashed', color=p[-1].get_color())
		plt.fill_between(x, mean-var, mean+var, alpha=0.2, color=p[-1].get_color())

	# plt.errorbar(np.array(model.hierarchies) + 1, mean, yerr=var, fmt='o')

	plt.xlabel('length')
	plt.ylabel('l2 difference')
	plt.title('diffence in Z')
	plt.legend()
	# plt.show()
	plt.savefig(metrics.OUT_DIR+'diff_in_z_%s'%(model.MODEL_CODE) + metrics.__get_timestamp() + '.png')


def eval_label_dist(model, validation_data):
	encodings_labels = metrics.__get_latent_reps(model.encoder, validation_data, model.MODEL_CODE)
	validation_data[:,:,-model.label_dim:] = 0
	n = validation_data.shape[0]
	encodings = metrics.__get_latent_reps(model.encoder, validation_data, model.MODEL_CODE)
	x = np.arange(model.timesteps) + 1
	diff = [[np.linalg.norm(encodings_labels[i,j] - encodings[i,j]) for j in range(model.timesteps)] for i in range(encodings.shape[0])]
	mean_diff = np.mean(diff, axis=0)
	std_diff = np.std(diff, axis=0)
	p = plt.plot(x, mean_diff, label='labels')
	plt.fill_between(x, mean_diff-std_diff, mean_diff+std_diff, alpha=0.3, color=p[-1].get_color())
	for h in tqdm(model.hierarchies):
		diff = [[np.linalg.norm(encodings_labels[i,h] - encodings_labels[i,hh]) for hh in range(model.timesteps)] for i in range(encodings.shape[0])]
		mean_diff = np.mean(diff, axis=0)
		std_diff = np.std(diff, axis=0)
		p = plt.plot(x, mean_diff, label=h+1)
		plt.fill_between(x, mean_diff-std_diff, mean_diff+std_diff, alpha=0.3, color=p[-1].get_color())

		a = np.random.choice(n, 1000, replace=False)
		b = np.random.choice(n, 1000, replace=False)
		diff = [[np.linalg.norm(encodings_labels[i,h] - encodings_labels[j,hh]) for hh in range(model.timesteps)] for i in a for j in b]
		mean_diff = np.mean(diff, axis=0)
		std_diff = np.std(diff, axis=0)
		plt.plot(x, mean_diff, linestyle='dashed', color=p[-1].get_color())
		plt.fill_between(x, mean_diff-std_diff, mean_diff+std_diff, alpha=0.2, color=p[-1].get_color())

	plt.legend()
	plt.show()

def eval_distance(model, validation_data):
	encodings = metrics.__get_latent_reps(model.encoder, validation_data, model.MODEL_CODE)
	# sum_z = np.sum(np.abs(encodings), axis=0)
	# print np.min(sum_z, axis=1)
	# print np.max(sum_z, axis=1)
	# for i in range(len(model.hierarchies)-1):
	# 	diff = [e[model.hierarchies[i+1]]-e[model.hierarchies[i]] for e in encodings]
	# 	plt.errorbar(np.arange(200), np.mean(diff, axis=0), yerr=np.std(diff, axis=0), label=i+1, fmt='o')

	# plt.legend()
	# plt.show()

		# print model.hierarchies[i], np.mean(diff), np.std(diff), np.mean(np.std(diff, axis=0)), np.std(np.std(diff, axis=0))

	def normalize(vec):
		return vec / np.linalg.norm(vec)

	for h in model.hierarchies[:-1]:
		skip_h = 5
		ind = range(0, validation_data.shape[0]-skip_h, skip_h)
		vect_h = [encodings[i+skip_h,h] - encodings[i,h] for i in ind]
		diff_30 = [np.linalg.norm(encodings[i,29] - encodings[i+skip_h,29]) for i in ind]
		diff_p_30 = [np.linalg.norm(encodings[i+skip_h,29] - encodings[i,29] - vect_h[k]) for k, i in enumerate(ind)]
		diff_i = [np.linalg.norm(e) for e in vect_h]

		diff_p_norm = [np.linalg.norm(normalize(encodings[i+skip_h,29]) - normalize(encodings[i,29] + vect_h[k])) for k, i in enumerate(ind)]

		# diff_i = [np.linalg.norm(encodings[i,h] - encodings[i+skip_h,h]) for i in ind]
		# diff_30 = [np.linalg.norm(encodings[i,29] - encodings[i+skip_h,29]) for i in ind]
		# diff_30 = [np.linalg.norm(validation_data[i] - validation_data[i+skip_h]) for i in ind]

		print np.mean(diff_p_30), np.mean(diff_p_norm)

		x_min, x_max = np.min(diff_30), np.max(diff_30)
		x = np.arange(x_min, x_max, (x_max-x_min)/50)

		f = np.poly1d(np.polyfit(diff_30,diff_p_norm, 2))
		p = plt.plot(x, f(x), label=h+1)
		plt.scatter(diff_30, diff_p_norm, alpha=0.2, c=p[-1].get_color())

		# f = np.poly1d(np.polyfit(diff_30,diff_i, 2))
		# p = plt.plot(x, f(x), label='ref. & pre.')
		# plt.scatter(diff_30, diff_i, alpha=0.5, c=p[-1].get_color())

		plt.xlabel('Difference l2 ref & ground truth')
		plt.ylabel('Difference l2 norm ground truth & norm prediction')
		plt.title('skip=%d, l=%d'%(skip_h, h))
		plt.legend()
		plt.show()

		x_min, x_max = np.min(diff_i), np.max(diff_i)
		x = np.arange(x_min, x_max, (x_max-x_min)/50)

		f = np.poly1d(np.polyfit(diff_i,diff_p_norm, 2))
		p = plt.plot(x, f(x), label=h+1)
		plt.scatter(diff_i, diff_p_norm, alpha=0.2, c=p[-1].get_color())

		# f = np.poly1d(np.polyfit(diff_30,diff_i, 2))
		# p = plt.plot(x, f(x), label='ref. & pre.')
		# plt.scatter(diff_30, diff_i, alpha=0.5, c=p[-1].get_color())

		plt.xlabel('Difference l2 partial sequences')
		plt.ylabel('Difference l2 norm ground truth & norm prediction')
		plt.title('skip=%d, l=%d'%(skip_h, h))
		plt.legend()
		plt.show()



	# norm_x = np.zeros(validation_data.shape[0])
	# norm_z = np.zeros(validation_data.shape[0])
	# for i, x in enumerate(tqdm(validation_data)):
	# 	norm_x[i] = np.linalg.norm(x[:2]-validation_data[0,:2])
	# 	norm_z[i] = np.linalg.norm(encodings[i,1]-encodings[0,1])

	# plt.scatter(norm_x, norm_z)
	# plt.xlabel('Pose norm')
	# plt.ylabel('Z norm')
	# plt.show()


### Evaluate generation/interpolation
# check in ave(x,y) belongs to the same subspace
# use nearest neighbour to compute score
def eval_generation(model, action_data, data_iterator, n=2, n_comp=2, cut=-1):
	ind_rand = np.random.choice(action_data.shape[0], n, replace=False)
	n = n/2
	encoded = metrics.__get_latent_reps(model.encoder, action_data[ind_rand], model.MODEL_CODE, n=cut)
	encoded = np.array([(encoded[i] + encoded[i+n])/2 for i in range(n)])
	action_data = metrics.__get_decoded_reps(model.decoder, encoded, model.MODEL_CODE)
	scores = np.array([[1000]*n]*len(model.labels))
	for xs,_ in data_iterator:
		x_idx = np.random.choice(len(xs), np.min(n_comp, len(xs)), replace=False)
		for x in xs[x_idx]:
			for i, z in enumerate(action_data):
				s = metrics.__pose_seq_error(z[:-model.label_dim], x[:-model.label_dim], cumulative=True)
				label_idx = np.argmax(x[-model.label_dim:])
				if scores[label_idx,i] > s:
					scores[label_idx,i] = s

	print 'mean', np.mean(scores, axis=0)
	print 'std', np.std(scores, axis=0)
	print 'max', np.max(scores, axis=0)
	print 'min', np.min(scores, axis=0)


def eval_center(model, action_data, n=200, n_comp=1000, cut=-1):
	encoded = metrics.__get_latent_reps(model.encoder, action_data, model.MODEL_CODE, n=cut)
	center_a = np.mean(encoded, axis=0)
	center_a = metrics.__get_decoded_reps(model.decoder, center_a, model.MODEL_CODE)
	center_raw = np.mean(action_data, axis=0)
	pseudo_center_idxs = np.random.choice(action_data.shape[0], n, replace=False)
	comp_idxs = np.random.choice(action_data.shape[0], n_comp, replace=False)
	scores = [[1000]*(n+2)]*n_comp
	for i in comp_idxs:
		for j in pseudo_center_idxs:
			scores[i][j] = metrics.__pose_seq_error(action_data[i], action_data[j], cumulative=True)
		score[i][-2] = metrics.__pose_seq_error(action_data[i], center_raw, cumulative=True)
		score[i][-1] = metrics.__pose_seq_error(action_data[i], center_a, cumulative=True)

	print 'mean', np.mean(scores, axis=0)
	print 'std', np.std(scores, axis=0)
	print 'min', np.min(scores, axis=0)
	print 'max', np.max(scores, axis=0)