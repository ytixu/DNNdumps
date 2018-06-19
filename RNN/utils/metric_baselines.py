# import matplotlib
# matplotlib.use('Agg')

from itertools import tee
import numpy as np
from tqdm import tqdm
import os.path
import glob
import metrics
import matplotlib.pyplot as plt

import fk_animate

LOAD_PATH = '../data/src/h3.6/results/'
_N = 8
_N_PRED = 25
_N_INPUT = 15
DATA_ITER_SIZE = 10000
RANDOM_N = 1000

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
			# fk_animate.animate_compare(gtp, pd)
		mean_err = np.mean(errors, axis=0)
		print basename, mean_err[[1,3,7,9,-1]]
		x = np.arange(pd.shape[0])
		plt.plot(x, mean_err, label=basename)

	plt.legend()
	plt.show()


def compare_raw_closest(from_path, data_iterator):
	import csv

	def load__(i, cut=0):
		if cut > 0:
			return {basename: [np.load(from_path + LOAD_PATH + basename + '_%d-%d.npy'%(i, j))[:cut] for j in range(_N)]
					for basename in iter_actions(from_path)}
		else:
			return {basename: [np.load(from_path + LOAD_PATH + basename + '_%d-%d.npy'%(i, j)) for j in range(_N)]
					for basename in iter_actions(from_path)}

	with open('../../results/nn_15_results.csv', 'wb') as csvfile:
		spamwriter = csv.writer(csvfile)
		# iter1, iter2 = tee(data_iterator)
		error_score = {basename:[10000]*_N for basename in iter_actions(from_path)}
		error_x = {basename:[None]*_N for basename in iter_actions(from_path)}
		gt = load__(0)
		gtp = load__(2, _N_PRED)
		pd = load__(1, _N_PRED)
		iterations = 0

		for xs, _ in data_iterator:
			idx = np.random.choice(xs.shape[0], min(RANDOM_N, xs.shape[0]), replace=False)
			for x in tqdm(xs[idx]):
				for basename in iter_actions(from_path):
					for i in range(_N):
						score = metrics.__pose_seq_error(x[:_N_INPUT], gt[basename][i][-_N_INPUT:])
						if score < error_score[basename][i]:
							error_score[basename][i] = score
							error_x[basename][i] = np.copy(x[_N_INPUT:])


			del xs
			iterations += 1
			print iterations
			# break

		for basename in iter_actions(from_path):
			error = [None]*_N
			error_ = [None]*_N
			for i in range(_N):
				error[i] = metrics.__pose_seq_error(error_x[basename][i], gtp[basename][i], cumulative=True)
				error_[i] = metrics.__pose_seq_error(pd[basename][i], gtp[basename][i], cumulative=True)
				np.save(from_path + LOAD_PATH + basename + '_nn_15-%d.npy'%i, error_x[basename][i])
				# fk_animate.animate_compare(gt[basename][i], gtp[basename][i],
				# 	error_x[basename][i], 'Nearest Neighbor (1/%d)'%(DATA_ITER_SIZE/RANDOM_N),
				# 	pd[basename][i], 'Residual sup. (MA)', from_path+LOAD_PATH+'images/')

			print basename
			_err = np.mean(error, axis=0)
			print 'nearest neighbor'
			print _err
			spamwriter.writerow([basename, 'Nearest nei. (1/%d)'%(DATA_ITER_SIZE/RANDOM_N)] + _err.tolist())
			_err = np.mean(error_, axis=0)
			print 'baseline error'
			print np.mean(error_, axis=0)
			spamwriter.writerow([basename, 'Residual sup. (MA)'] + _err.tolist())


def plot_results(plot_csv):
	import csv
	with open(plot_csv, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		basename = ''
		for row in reader:
			print row[0]
			if basename != row[0] and basename != '':
				plt.legend()
				plt.xlabel('time-steps')
				plt.ylabel('error')
				plt.title(basename)
				plt.savefig('../../results/nearest-nei/%s-1-10.png'%(basename))
				plt.close()

			basename = row[0]
			plt.plot(range(1,len(row)-1), map(float, row[2:]), label=row[1])

def plot_results_npy(from_path, npy_files_dirs, method_names):
	for basename in iter_actions(from_path):
		for i in range(len(method_names)):
			pd = np.load(npy_files_dirs[i] + basename + '.npy')
			score = [None]*_N
			score_ = [None]*_N
			t = pd.shape[1]

			for j in range(_N):
				gt = np.load(from_path + LOAD_PATH + basename + '_2-%d.npy'%j)[:t]
				score[j] = metrics.__pose_seq_error(pd[j], gt, cumulative=True)
				if i == 0:
					pd_b = np.load(from_path + LOAD_PATH + basename + '_1-%d.npy'%j)[:t]
					score_[j] = metrics.__pose_seq_error(pd_b, gt, cumulative=True)

			plt.plot(range(1,t+1), np.mean(score, axis=0), label=method_names[i])

			if i == 0:
				plt.plot(range(1,t+1), np.mean(score_, axis=0), label='Residual sup. (MA)')

		plt.legend()
		plt.xlabel('time-steps')
		plt.ylabel('error')
		plt.title(basename)
		plt.savefig('../../new_out/L-RNN-%s.png'%(basename))
		plt.close()

def animate_results(from_path, predict, predict_name, baseline='1',
		baseline_name='Residual sup. (MA)', ground_truth='2'):
	for basename in iter_actions(from_path):
		for i in range(_N):
			print basename, i
			gt = np.load(from_path + LOAD_PATH + basename + '_0-%d.npy'%i)
			gtp = np.load(from_path + LOAD_PATH + basename + '_%s-%d.npy'%(ground_truth,i))
			bpd = np.load(from_path + LOAD_PATH + basename + '_%s-%d.npy'%(baseline, i))
			pd = np.load(from_path + LOAD_PATH + basename + '_%s-%d.npy'%(predict, i))

			fk_animate.animate_compare(gt, gtp[:len(pd)], pd, predict_name, bpd[:len(pd)], baseline_name,
					from_path+LOAD_PATH+'images/%s-%d-%s'%(basename, i, predict_name.replace('.', '').replace('/', '-').replace(' ', '-')))


def compare_label_embedding(model, nn, data_iterator, with_label=True):

	#for x, _ in data_iterator:
	#	enc = model.encoder.predict(x[:2])[:, model.hierarchies[-2:]]
	#	enc1 = nn.model.predict(enc[:,0])
	#	print np.mean(np.abs(enc1-enc[:,1]))

	#return

	import image
	# embedding = metrics.get_label_embedding(model, data_iterator, without_label_only=(not with_label), subspaces=model.hierarchies[-2:])
	# mean_diff, diff = metrics.get_embedding_diffs(embedding[:,1], embedding[:,0])
	cut = model.hierarchies[-2]+1
	pred_n = model.timesteps-cut
	for basename in iter_actions():
		print basename
		pose_ref = np.zeros((_N, model.timesteps, model.input_dim))
		pose_pred_bl = np.zeros((_N, pred_n, model.input_dim-model.label_dim))
		pose_gt = np.zeros((_N, pred_n, model.input_dim-model.label_dim))

		for i in tqdm(range(_N)):
			gt = np.load(LOAD_PATH + basename + '_0-%d.npy'%i)
			pd = np.load(LOAD_PATH + basename + '_1-%d.npy'%i)
			gtp = np.load(LOAD_PATH + basename + '_2-%d.npy'%i)

			pose_ref[i,:cut,:-model.label_dim] = gt[-cut:]
			pose_pred_bl[i] = pd[:pred_n]
			pose_gt[i] = gtp[:pred_n]

			pose_ref[i,cut:,:-model.label_dim] = gtp[:pred_n]

		if with_label:
			print model.labels[basename]
			pose_ref[:,:,-model.label_dim:] = model.labels[basename]

		enc = model.encoder.predict(pose_ref)
		print enc.shape, pose_ref.shape
		#new_enc = nn.model.predict(enc[:,cut-1])
		#print np.mean(np.abs(new_enc - enc[:,-1]))

		dec = model.decoder.predict(enc[:,-1])
		print dec.shape
		image.plot_poses(dec, title='dec', image_dir='../new_out/')
		continue
		# new_enc = model.encoder.predict(pose_ref)[:,cut-1] + mean_diff
		# # pose_pred = model.decoder.predict(new_enc)
		pose_pred = model.decoder.predict(new_enc)[:,-pred_n:,:-model.label_dim]
		error_bl = [metrics.__pose_seq_error(pose_gt[i], pose_pred_bl[i]) for i in range(_N)]
		error = [metrics.__pose_seq_error(pose_gt[i], pose_pred[i]) for i in range(_N)]
		print error
		print error_bl
		# image.plot_poses(pose_pred, title='rnn', image_dir='../new_out/')
		# image.plot_poses(pose_pred_bl, title='baseline', image_dir='../new_out/')
		# image.plot_poses(pose_gt, title='gt', image_dir='../new_out/')
		# np.save('../new_out/LRNN-%s.npy'%basename, pose_pred)

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
		np.save('../new_out/HRNN-%s.npy'%basename, pose_pred)


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
	#get_baselines('../')
	#
	# import parser
	# data_iterator = parser.data_generator('../../data/h3.6/train/', '../../data/h3.6/train/', _N_PRED+_N_INPUT, DATA_ITER_SIZE)
	# compare_raw_closest('../', data_iterator)

	# plot_results('../../results/nn_15_results.csv')
	animate_results('../', 'nn_15', 'Nearest nei. (1/10)')

	# plot_results_npy('../', ['../../new_out/H_GRU/raw-t30-l300/HRNN-', '../../new_out/L_RNN-t30-l400/without-labels/LRNN-'], ['H-RNN', 'L-RNN'])
