from itertools import tee
import numpy as np
from tqdm import tqdm
import os.path
import glob
import metrics
import matplotlib.pyplot as plt

LOAD_PATH = '../../data/src/h3.6/results/'
_N = 8
DATA_ITER_SIZE = 100


def iter_actions(from_path=''):
	for filename in glob.glob(from_path+LOAD_PATH + '*_0-0.npy'):
		yield os.path.basename(filename).split('_')[0]

def get_baselines(from_path=''):
	# import image
	for basename in iter_actions(from_path):
		errors = []
		for i in range(8):
			gt = np.load(from_path + LOAD_PATH + basename + '_0-%d.npy'%i)
			gtp = np.load(from_path + LOAD_PATH + basename + '_2-%d.npy'%i)
			pd = np.load(from_path + LOAD_PATH + basename + '_1-%d.npy'%i)
			if len(errors) == 0:
				errors = np.zeros((8, gtp.shape[0]))
			for j in range(gtp.shape[0]):
				errors[i, j] = metrics.__pose_error(gtp[j], pd[j])
			# print basename
			# image.plot_poses([gt[-10:], gtp[:10]])
		mean_err = np.mean(errors, axis=0)
		print basename, mean_err[[1,3,7,9,-1]]
		x = np.arange(pd.shape[0])
		plt.plot(x, mean_err, label=basename)

	plt.legend()
	plt.show()

def compare_raw_closest(data_iterator):
	iter1, iter2 = tee(data_iterator)
	for basename in iter_actions():
		print basename
		error = [None]*_N
		error_ = [None]*_N
		for i in tqdm(range(_N)):
			gt = np.load(LOAD_PATH + basename + '_0-%d.npy'%i)
			pd = np.load(LOAD_PATH + basename + '_1-%d.npy'%i)
			gtp = np.load(LOAD_PATH + basename + '_2-%d.npy'%i)

			best_score = 10000
			best_x = None
			n = gt.shape[0]
			for xs, _ in iter1:
				for x in tqdm(xs):
					score = metrics.__pose_seq_error(x[:n], gt)
					if score < best_score:
						best_score = score
						best_x = x[n:]
				break
			iter1, iter2 = tee(iter2)
			error[i] = metrics.__pose_seq_error(best_x, gtp, cumulative=True)
			error_[i] = metrics.__pose_seq_error(pd, gtp, cumulative=True)

		print error
		print error_


def compare_label_embedding(model, data_iterator):
	import image
	embedding = metrics.get_label_embedding(model, data_iterator, without_label_only=True, subspaces=model.hierarchies[-2:])
	mean_diff, diff = get_embedding_diffs(embedding[1], embedding[0])
	# std_diff = np.std(diff, axis=0)
	cut = model.hierarchies[-2]+1
	pred_n = model.timesteps-cut
	for basename in iter_actions():
		print basename
		pose_ref = np.zeros((_N, model.timesteps, model.input_dim))
		pose_pred_bl = np.zeros((_N, model.timesteps-cut, model.input_dim-model.label_dim))
		pose_gt = np.zeros((_N, model.timesteps-cut, model.input_dim-model.label_dim))

		for i in tqdm(range(_N)):
			gt = np.load(LOAD_PATH + basename + '_0-%d.npy'%i)
			pd = np.load(LOAD_PATH + basename + '_1-%d.npy'%i)
			gtp = np.load(LOAD_PATH + basename + '_2-%d.npy'%i)

			pose_ref[_N,:cut,:-model.label_dim] = gt[-cut:]
			pose_pred_bl[_N] = pd[:pred_n]
			pose_gt = gtp[:pred_n]

		new_enc = model.encoder.predict(pose_ref)[:,cut-1] + mean_diff
		pose_pred = model.decoder.predict(new_enc)[:,:,:-model.label_dim]
		error_bl = [__pose_seq_error(pose_gt[i], pose_pred_bl[i]) for i in range(_N)]
		error = [__pose_seq_error(pose_gt[i], pose_pred[i]) for i in range(_N)]
		# error_0_vel =
		print np.mean(error), np.mean(error_bl)
		image.plot_poses(pose_pred)

def compare_embedding(model, data_iterator):
	import image
	embedding = metrics.get_embedding(model, data_iterator, subspace=model.hierarchies[-2:])
	mean_diff, diff = metrics.get_embedding_diffs(embedding[1], embedding[0])
	std_diff = np.std(diff, axis=0)
	cut = model.hierarchies[-2]+1
	pred_n = model.timesteps-cut
	for basename in iter_actions():
		print basename
		n = 8
		pose_ref = np.zeros((n, model.timesteps, model.input_dim))
		pose_pred_bl = np.zeros((n, model.timesteps-cut, model.input_dim))
		pose_gt = np.zeros((n, model.timesteps-cut, model.input_dim))

		for i in tqdm(range(n)):
			gt = np.load(LOAD_PATH + basename + '_0-%d.npy'%i)
			pd = np.load(LOAD_PATH + basename + '_1-%d.npy'%i)
			gtp = np.load(LOAD_PATH + basename + '_2-%d.npy'%i)

			pose_ref[i,:cut] = gt[-cut:]
			pose_pred_bl[i] = pd[:pred_n]
			pose_gt[i] = gtp[:pred_n]

		new_enc = model.encoder.predict(pose_ref)[:,cut-1] + mean_diff
		pose_pred = model.decoder.predict(new_enc)[:,:pred_n]
		error_bl = [metrics.__pose_seq_error(pose_gt[i], pose_pred_bl[i]) for i in range(n)]
		error = [metrics.__pose_seq_error(pose_gt[i], pose_pred[i]) for i in range(n)]
		print np.mean(error), np.mean(error_bl)
		image.plot_poses(pose_pred, title='rnn')
		image.plot_poses(pose_pred_bl, title='baseline')
		image.plot_poses(pose_gt, title='gt')


def compare(model, data_iterator):
	import image
	h = model.timesteps
	embedding = metrics.get_embedding(model, data_iterator, model.timesteps-1)
	# methods = ['Closest', 'Mean-45', 'Mean-75', 'Mean-100', 'Random-5000', 'Random-10000', 'Multi']
	methods = ['Add', 'Add-Closest', 'Add-Mean-30', 'Add-Mean-45', 'Add-Mean-75']
	# methods = ['Closest', 'Mean-30', 'Mean-45', 'Random-5000', 'Random-10000']
	for k, cut in enumerate(model.hierarchies[-2:-1]):
		# k = (h-cut)/3
		errors = {}
		errors_ = {}
		for basename in iter_actions():
			errors[basename] = []
			errors_[basename] = []
			print basename
			for i in tqdm(range(8)):
				gt = np.load(LOAD_PATH + basename + '_0-%d.npy'%i)
				pd = np.load(LOAD_PATH + basename + '_1-%d.npy'%i)
				gtp = np.load(LOAD_PATH + basename + '_2-%d.npy'%i)
				if model.MODEL_CODE == metrics.HL_LSTM:
					l = np.zeros((gt.shape[0], model.label_dim))
					l[:,model.labels[basename]] = 1
					gt = np.concatenate((gt, l), axis=1)

				# pose = metrics.__get_next_half(embedding, gt[-10:], encoder, decoder, h, model_name)
				# poses = metrics.__get_consecutive(embedding, gt[-h:], model, cut+1, k)
				# poses = metrics.__get_consecutive_multi(embedding, gt[-h:], model, cut+1, k)
				poses = metrics.__get_consecutive_add(embedding, gt[-h:], model, cut+1)
				ground_truth = metrics.__autoencode(model.autoencoder, np.array([gtp[:h]]), model.MODEL_CODE)
				ground_truth = np.reshape(ground_truth, (-1, model.timesteps, model.input_dim))[-1]
				# pose_ref = poses[0]
				# poses = poses[1:]
				n = poses.shape[1]
				# metrics.__get_embedding_path(gt, encoder, decoder, h, model_name)
				if len(errors[basename]) == 0:
					errors[basename] = np.zeros((8, poses.shape[0], n))
					errors_[basename] = np.zeros((8, 4, n))
				for j in range(n):
					for k, p in enumerate(poses):
						errors[basename][i, k, j] = metrics.__pose_seq_error(gtp[:j+1], p[:j+1])
					errors_[basename][i, 3, j] = metrics.__pose_seq_error(gtp[:j+1], pd[:j+1])

				errors_[basename][i,0,:] = metrics.__zeros_velocity_error(gt[-n:], gtp[:n])[:]
				errors_[basename][i,1,:] = metrics.__average_2_error(gt[-n:], gtp[:n])[:]
				errors_[basename][i,2,:] = metrics.__average_4_error(gt[-n:], gtp[:n])[:]

				# image.plot_poses([gt[-h:], gtp[:h]], [pose[:h], pd[:h]])
				best_error = np.min(errors[basename][i, :,-1])
				b_error = errors_[basename][i,-1,-1]
				image_dir = '../results/t30-l200/low_error/refined-add/'
				if best_error > errors_[basename][i,-1,-1]:
					image_dir = '../results/t30-l200/high_error/refined-add/'

				image.plot_poses([gt[-n:], gtp[:n], ground_truth[:n]], np.concatenate([poses, [pd[:n]]], axis=0),
					title='%6.3f-%6.3f (%s - C, M, R-5000-10000, B) %d-%d'%(b_error, best_error/b_error, basename, cut, k), image_dir=image_dir)
			# print basename, mean_err[[1,3,7,9,-1]]
			# if basename in ['walking', 'eating', 'smoking', 'discussion']:
			x = np.arange(n)+1
			for k, method in enumerate(methods):
				mean_err = np.mean(errors[basename][:,k,:], axis=0)
				plt.plot(x, mean_err, label=method)
			for k, method in enumerate(['0 velocity', 'avg-2', 'avg-4', 'baseline']):
				mean_err_ = np.mean(errors_[basename][:,k,:], axis=0)
				plt.plot(x, mean_err_, '--', label=method)

			plt.legend()
			plt.title('%s-%d'%(basename, cut))
			# plt.show()
			plt.savefig('../results/t30-l200/graphs/refined-add/%s-%d-%d.png'%(basename, cut, k))
			plt.close()


		errors = np.reshape(np.array(errors.values()), (-1, poses.shape[0], n))
		errors_ = np.reshape(np.array(errors_.values()), (-1, 4, n))
		for k, method in enumerate(methods):
			mean_err = np.mean(errors[:,k,:], axis=0)
			plt.plot(x, mean_err, label=method)
		for k, method in enumerate(['0 velocity', 'avg-2', 'avg-4', 'baseline']):
			mean_err_ = np.mean(errors_[:,k,:], axis=0)
			plt.plot(x, mean_err_, '--', label=method)

		plt.legend()
		plt.title('Total-%d'%(cut))
		# plt.show()
		plt.savefig('../results/t30-l200/graphs/refined-add/total-%d-%d.png'%(cut, k))
		plt.close()



if __name__ == '__main__':
	# get_baselines('../')
	import parser
	data_iterator = parser.data_generator('../../data/h3.6/train/', '../../data/h3.6/train/', 149, DATA_ITER_SIZE)
	print data_iterator
	compare_raw_closest(data_iterator)
