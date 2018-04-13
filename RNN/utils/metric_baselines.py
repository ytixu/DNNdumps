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

def compare(encoder, decoder, h, model_name):
	import image
	for basename in iter_actions():
		errors = []
		errors_ = []
		for i in range(8):
			gt = np.load(LOAD_PATH + basename + '_0-%d.npy'%i)
			pd = np.load(LOAD_PATH + basename + '_1-%d.npy'%i)
			gtp = np.load(LOAD_PATH + basename + '_2-%d.npy'%i)
			# pose = metrics.__get_next_half(embedding, gt[-10:], encoder, decoder, h, model_name)
			pose = metrics.__get_consecutive(gt[-h:], encoder, decoder, h, model_name, n=3)
			# metrics.__get_embedding_path(gt, encoder, decoder, h, model_name)
			if len(errors) == 0:
				errors = np.zeros((8, pose.shape[0]))
				errors_ = np.zeros((8, pose.shape[0]))
			for j in range(pose.shape[0]):
				errors[i, j] = metrics.__pose_error(gtp[j], pose[j])
				errors_[i, j] = metrics.__pose_error(gtp[j], pd[j])
			image.plot_poses([gt[-h:], gtp[:h]], [pose[:h], pd[:h]])
		mean_err = np.mean(errors, axis=0)
		# print basename, mean_err[[1,3,7,9,-1]]
		# if basename in ['walking', 'eating', 'smoking', 'discussion']:
		x = np.arange(pose.shape[0])
		plt.plot(x, mean_err, label=basename)
		mean_err_ = np.mean(errors_, axis=0)
		plt.plot(x, mean_err_, '--', label=basename)

		plt.legend()
		plt.show()

if __name__ == '__main__':
	get_baselines('../')