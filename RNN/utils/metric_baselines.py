import numpy as np
from tqdm import tqdm
import os.path
import glob
import metrics
import matplotlib.pyplot as plt

LOAD_PATH = '../data/src/h3.6/results/'

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

def compare(model, data_iterator):
	import image
	h = model.timesteps
	embedding = metrics.get_embedding(model, data_iterator, model.timesteps-1)
	for basename in iter_actions():
		errors = []
		errors_ = []
		for i in range(8):
			gt = np.load(LOAD_PATH + basename + '_0-%d.npy'%i)
			pd = np.load(LOAD_PATH + basename + '_1-%d.npy'%i)
			gtp = np.load(LOAD_PATH + basename + '_2-%d.npy'%i)
			if model.MODEL_CODE in [metrics.HL_LSTM, metrics.Prior_LSTM]:
				l = np.zeros((gt.shape[0], model.label_dim))
				l[:,model.labels[basename]] = 1
				gt = np.concatenate((gt, l), axis=1)
				print gt.shape

			# pose = metrics.__get_next_half(embedding, gt[-10:], encoder, decoder, h, model_name)
			poses = metrics.__get_consecutive(embedding, gt[-h:], model)
			n = poses.shape[1]
			# metrics.__get_embedding_path(gt, encoder, decoder, h, model_name)
			if len(errors) == 0:
				errors = np.zeros((8, poses.shape[0], n))
				errors_ = np.zeros((8, 4, n))
			for j in range(n):
				for k, p in enumerate(poses):
					errors[i, k, j] = metrics.__pose_seq_error(gtp[:j+1], p[:j+1])
				errors_[i, 3, j] = metrics.__pose_seq_error(gtp[:j+1], pd[:j+1])
			
			errors_[i,0,:] = metrics.__zeros_velocity_error(gt[-n:], gtp[:n])[:]
			errors_[i,1,:] = metrics.__average_2_error(gt[-n:], gtp[:n])[:]
			errors_[i,2,:] = metrics.__average_4_error(gt[-n:], gtp[:n])[:]

			# image.plot_poses([gt[-h:], gtp[:h]], [pose[:h], pd[:h]])
			best_error = np.min(errors[i, :,-1])
			b_error = errors_[i,-1,-1]
			image_dir = '../results/t10-l100/low_error/'
			if best_error > errors_[i,-1,-1]:
				image_dir = '../results/t10-l100/high_error/'

			image.plot_poses([gt[-n:], gtp[:n]], np.concatenate([poses, [pd[:n]]], axis=0), 
				title='%6.3f-%6.3f (%s - C, M, R-5000-10000, B)'%(b_error, best_error/b_error, basename), image_dir=image_dir)
		# print basename, mean_err[[1,3,7,9,-1]]
		# if basename in ['walking', 'eating', 'smoking', 'discussion']:
		x = np.arange(n)+1
		for k, method in enumerate(['Closest', 'Mean', 'Random-5000', 'Random-10000']):
			mean_err = np.mean(errors[:,k,:], axis=0)
			plt.plot(x, mean_err, label=method)
		for k, method in enumerate(['0 velocity', 'avg-2', 'avg-4', 'baseline']):
			mean_err_ = np.mean(errors_[:,k,:], axis=0)
			plt.plot(x, mean_err_, '--', label=method)

		plt.legend()
		plt.title(basename)
		# plt.show()
		plt.savefig('../results/t10-l100/graphs/' + basename + '.png') 
		plt.close()

if __name__ == '__main__':
	get_baselines('../')