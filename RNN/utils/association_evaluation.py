import matplotlib
matplotlib.use('Agg')

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

import metrics

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


def __save_score(scores, model, name):
	labels = {v: k for k, v in model.labels.iteritems()}
	with open('../new_out/%s-%s-t%d-l%d.json'%(name, model.NAME, model.timesteps, model.latent_dim), 'wb') as jsonfile:
		json.dump({
			'labels':labels,
			'mean':np.mean(scores, axis=0).tolist(),
			'std':np.std(scores, axis=0).tolist(),
			'max':np.max(scores, axis=0).tolist(),
			'min':np.min(scores, axis=0).tolist()
		}, jsonfile)

	print labels
	print 'mean', np.mean(scores, axis=0)
	print 'std', np.std(scores, axis=0)
	print 'min', np.min(scores, axis=0)
	print 'max', np.max(scores, axis=0)

### Evaluate generation/interpolation
# check in ave(x,y) belongs to the same subspace
# use nearest neighbour to compute score
def eval_generation(model, action_data, data_iterator, n=200, n_comp=1000, cut=-1):
	if cut == -1:
		cut = model.hierarchies[-1]
	ind_rand = np.random.choice(action_data.shape[0], n, replace=False)
	n = n/2
	encoded = metrics.__get_latent_reps(model.encoder, action_data[ind_rand], model.MODEL_CODE, n=cut)
	encoded = np.array([(encoded[i] + encoded[i+n])/2 for i in range(n)])
	encoded = metrics.__get_latent_reps(model.encoder, action_data[ind_rand], model.MODEL_CODE, n=cut)
	encoded = np.array([(encoded[i] + encoded[i+n])/2 for i in range(n)])
	action_data = metrics.__get_decoded_reps(model.decoder, encoded, model.MODEL_CODE)
	scores = [[1000.0]*len(model.labels)]*n
	count = 0
	for xs,_ in data_iterator:
		x_idx = np.random.choice(xs.shape[0], min(n_comp, xs.shape[0]), replace=False)
		for x in tqdm(xs[x_idx]):
			for i, z in enumerate(action_data):
				s = metrics.__pose_seq_error(z[:-model.label_dim], x[:-model.label_dim])
				label_idx = np.argmax(x[-model.label_dim:]) - model.input_dim + model.label_dim
				if scores[i][label_idx] > s:
					scores[i][label_idx] = s
		del xs
		print count
		count += 1

	__save_score(scores, model, 'eval_generation')

## Check if encoded center is as close to others as in the raw space
def eval_center(model, action_data, action_name, n=200, n_comp=1000, cut=-1):
	LABEL_GEN_CENTERS = '../new_out/L_RNN-t30-l400/generate_from_labels/eval_generation_from_label-gen_poses-L_GRU.npy'
	if cut == -1:
		cut = model.hierarchies[-1]

	encoded = metrics.__get_latent_reps(model.encoder, action_data, model.MODEL_CODE, n=cut)
	center_a = np.array([np.mean(encoded, axis=0)])
	center_a = metrics.__get_decoded_reps(model.decoder, center_a, model.MODEL_CODE)[0]
	center_raw = np.mean(action_data, axis=0)
	pseudo_center_idxs = np.random.choice(action_data.shape[0], n, replace=False)
	comp_idxs = np.random.choice(action_data.shape[0], min(n_comp, action_data.shape[0]), replace=False)
	scores = np.array([[1000.0]*(n+3)]*n_comp)
	print scores.shape

	if model.MODEL_CODE == metrics.L_LSTM:
		center_a = center_a[:,:-model.label_dim]
		center_raw = center_raw[:,:-model.label_dim]
		action_data = action_data[:,:,:-model.label_dim]

	center_from_label = np.load(LABEL_GEN_CENTERS)[model.labels[action_name]]

	for l, i in enumerate(tqdm(comp_idxs)):
		for k, j in enumerate(pseudo_center_idxs):
			scores[l][k] = metrics.__pose_seq_error(action_data[i], action_data[j])
		scores[l][-3] = metrics.__pose_seq_error(action_data[i], center_from_label)
		scores[l][-2] = metrics.__pose_seq_error(action_data[i], center_a)
		scores[l][-1] = metrics.__pose_seq_error(action_data[i], center_raw)

	__save_score(scores, model, 'eval_center')
	print 'animating...'
	import fk_animate
	fk_animate.animate_motion(center_raw, 'center raw', '../new_out/center_raw_animate-%s-t%d-l%d'%(model.NAME, model.timesteps, model.latent_dim))
	fk_animate.animate_motion(center_a, 'center latent', '../new_out/center_latent_animate-%s-t%d-l%d'%(model.NAME, model.timesteps, model.latent_dim))

## generate motion from label
# which match with the center
def eval_generation_from_label(model, data_iterator, cut=-1):
	if cut == -1:
		cut = model.hierarchies[-1]
	diff_by_label = np.zeros((len(model.labels), model.latent_dim))
	diff_n = [0]*len(model.labels)
	count = 0
	for xs, _ in data_iterator:
		count += 1
		print count

		z_motion = metrics.__get_latent_reps(model.encoder, xs, model.MODEL_CODE, n=cut)
		idx = np.argmax(xs[:,0,-model.label_dim:], axis=1)
		# print z_motion.shape, idx.shape
		# print idx
		xs[:,:,:-model.label_dim] = 0
		z_label = metrics.__get_latent_reps(model.encoder, xs, model.MODEL_CODE, n=cut)
		diff = z_motion - z_label
		for i in range(model.label_dim):
			idx_label = np.where(idx == i)
			# print diff[idx_label].shape
			n_d = diff[idx_label].shape[0]
			if n_d > 0:
				# print np.sum(diff[idx_label], axis=0).shape
				diff_by_label[i] = diff_by_label[i] + np.sum(diff[idx_label], axis=0)
				diff_n[i] = diff_n[i] + n_d
		del xs, idx, diff

	x = range(model.latent_dim)
	for i, n_d in enumerate(diff_n):
		diff_by_label[i] = diff_by_label[i]/n_d
		plt.plot(x, diff_by_label[i])
	plt.xlabel('latent dimensions')
	plt.ylabel('mean difference')
	plt.title('Mean difference with/without motion')
	plt.savefig('../new_out/eval_generation_from_label-z-'+model.NAME+'.png')
	plt.close()

	print 'generating...'
	valid_data = np.zeros((model.label_dim, model.timesteps, model.input_dim))
	for i in range(model.timesteps):
		valid_data[:,i,-model.label_dim:] = np.identity(model.label_dim)
	z_label = metrics.__get_latent_reps(model.encoder, valid_data, model.MODEL_CODE, n=cut)
	z_gen = z_label + diff_by_label
	# action_pred = metrics.__get_decoded_reps(model.decoder, z_gen, model.MODEL_CODE)
	# filename = '../new_out/eval_generation_from_label-gen_poses-'+model.NAME+'.npy'
	# np.save(filename, action_pred[:,:,:-model.label_dim])
	filename = '../new_out/eval_generation_from_label-gen_z-'+model.NAME+'.npy'
	np.save(filename, z_gen)

	# print 'animating...'
	# animate_poses(filename, model, '../new_out/eval_generation_from_label-animate-')

def transfer_motion(model, action_data, from_motion_name, to_motion_name, data_iterator, n=10, n_comp=1000, cut=-1):
	LABEL_GEN_Z = '../new_out/L_RNN-t30-l400/generate_from_labels/eval_generation_from_label-gen_z-L_GRU.npy'
	LABEL_GEN_CENTERS = '../new_out/L_RNN-t30-l400/generate_from_labels/eval_generation_from_label-gen_poses-L_GRU.npy'
	if cut == -1:
		cut = model.hierarchies[-1]

	action_data = action_data[np.random.choice(action_data.shape[0], n, replace=False)]
	z_actions = metrics.__get_latent_reps(model.encoder, action_data, model.MODEL_CODE, n=cut)

	z_labels = np.load(LABEL_GEN_Z)[[model.labels[from_motion_name], model.labels[to_motion_name]]]
	z_infered = z_actions - z_labels[0] + z_labels[1]
	action_from_z = metrics.__get_decoded_reps(model.decoder, z_infered, model.MODEL_CODE)
	# action_normalized = metrics.__get_decoded_reps(model.decoder, z_infered/np.linalg.norm(z_infered), model.MODEL_CODE)

	center_from_label = np.zeros((2, model.timesteps, model.input_dim))
	center_from_label[:,:,:-model.label_dim] = np.load(LABEL_GEN_CENTERS)[[model.labels[from_motion_name], model.labels[to_motion_name]]]
	center_from_label[0,:,-model.label_dim+model.labels[from_motion_name]] = 1
	center_from_label[1,:,-model.label_dim+model.labels[to_motion_name]] = 1
	z_labels = metrics.__get_latent_reps(model.encoder, center_from_label, model.MODEL_CODE, n=cut)
	z_infered = z_actions - z_labels[0] + z_labels[1]
	action_from_pose = metrics.__get_decoded_reps(model.decoder, z_infered, model.MODEL_CODE)

	print 'animating...'
	import fk_animate
	save_path = '../new_out/transfer_motion-%s-to-%s-'%(from_motion_name, to_motion_name)
	for i in range(z_actions.shape[0]):
		fk_animate.animate_motion([action_data[i], action_from_z[i], action_from_pose[i]], [from_motion_name, to_motion_name, to_motion_name+'(+name)'], save_path+str(i))
		# fk_animate.animate_motion([action_data[i], action_from_z[i], action_normalized[i]], [from_motion_name, to_motion_name, to_motion_name+'(norm)'], save_path+str(i)+'-')

	scores = [[1000.0]*len(model.labels)]*n
	count = 0
	for xs,_ in data_iterator:
		x_idx = np.random.choice(xs.shape[0], min(n_comp, xs.shape[0]), replace=False)
		for x in tqdm(xs[x_idx]):
			for i, z in enumerate(action_from_z):
				s = metrics.__pose_seq_error(z[:-model.label_dim], x[:-model.label_dim])
				label_idx = np.argmax(x[-model.label_dim:]) - model.input_dim + model.label_dim
				if scores[i][label_idx] > s:
					scores[i][label_idx] = s
		del xs
		print count
		count += 1

	print scores
	np.save(save_path+'scores.npy', scores)

def plot_transfer_motion(model, filename):
	data = np.load(filename)
	name = filename.split('transfer_motion-')[-1].split('-scores')[0]
	x = range(model.label_dim)
	for i in range(data.shape[0]):
		plt.scatter(x, data[i])

	labels = {v: k for k, v in model.labels.iteritems()}
	plt.xticks(x, [labels[i] for i in range(model.label_dim)], rotation='vertical')
	plt.xlabel('category')
	plt.ylabel('closest distance')
	plt.margins(0.1)
	plt.subplots_adjust(bottom=0.25)
	plt.title('Distance to trans. motion (%s)'%name)
	plt.savefig('../new_out/plot_transfer_motion-'+name+'.png')
	plt.close()


def plot_results(directory, model_name, action_type):
	with open(directory+'eval_generation-'+model_name+'.json', 'rb') as jsonfile:
		data = json.loads(jsonfile.read())
		x = range(len(data['labels']))
		y = data['mean']
		yerr = [data['min'], data['max']]
		plt.errorbar(x, y, yerr=yerr, fmt='o')
		yerr = data['std']
		plt.errorbar(x, y, yerr=yerr, fmt='o')
		plt.xticks(x, [data['labels'][str(i)] for i in range(len(x))], rotation='vertical')
		plt.xlabel('category')
		plt.ylabel('closest distance')
		plt.margins(0.1)
		plt.subplots_adjust(bottom=0.25)
		plt.title('Distance to interpolated %s motion (min, mean, std, max)'%action_type)
		plt.savefig(directory+'eval_generation-'+model_name+'-std.png')
		plt.close()


	with open(directory+'eval_center-'+model_name+'.json', 'rb') as jsonfile:
		data = json.loads(jsonfile.read())
		x = range(200)+[210, 220, 230]
		idx = np.flip(np.argsort(data['mean'][:-3]), 0)
		y = [data['mean'][i] for i in idx]+data['mean'][-3:]
		print len(x), len(y)
		yerr = [[data['min'][i] for i in idx]+data['min'][-3:],
			[data['max'][i] for i in idx]+data['max'][-3:]]
		plt.errorbar(x, y, yerr=yerr, fmt='o')
		yerr = [data['std'][i] for i in idx]+data['std'][-3:]
		plt.errorbar(x, y, yerr=yerr, fmt='o')
		plt.xticks(x, ['']*200 + ['gen-label', 'z', 'raw'], rotation='vertical')
		plt.subplots_adjust(bottom=0.15)
		plt.margins(0.1)
		plt.xlabel('centers')
		plt.ylabel('distance to other %s motions'%action_type)
		plt.title('Centers comparison for %s (min, mean, max)'%action_type)
		plt.savefig(directory+'eval_center-'+model_name+'-std.png')
		plt.close()

def animate_poses(filename, model, save_path):
	import fk_animate
	labels = {v: k for k, v in model.labels.iteritems()}
	poses = np.load(filename)
	for i in tqdm(range(model.label_dim)):
		fk_animate.animate_motion(poses[i], labels[i], save_path)


if __name__ == '__main__':
	action_type = 'sitting'
	plot_results('../../new_out/'+action_type+'/', 'L_GRU-t30-l400', action_type)
