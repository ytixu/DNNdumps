import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime

def plot_poses(batch, batch2=[], title='Poses', args='', image_dir=None):
	timesteps = len(batch[0])
	n = len(batch)
	n_total = len(batch) + len(batch2)
	skip_n = timesteps
	skip_t = timesteps/skip_n
	f, axarr = plt.subplots(n_total, skip_n, sharex=True, sharey=True)

	for i in range(n):
		new_x = np.reshape(batch[i], (timesteps, -1, 3))
		for j in range(skip_n):
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[i, j], (new_x[j*skip_t])[l], c, 1)

	for i in range(len(batch2)):
		new_x = np.reshape(batch2[i], (timesteps, -1, 3))
		for j in range(skip_n):
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[i+n, j], (new_x[j*skip_t])[l], c, 2)


	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
	# plt.show()
	if image_dir == None:
		image_dir = '../out/'
	f.savefig(image_dir + title.lower().replace(' ', '_')+args+ strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png', dpi=192)
	plt.close(f)