import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime


def plot_data(data):
	for x in data:
		plt.plot(range(x.shape[0]), x)
	plt.savefig('data-exp-viz.png', dpi=192)
        plt.close()

M_POSE_LINES = {'r':[0,1,2,3],
			'g':[0,6,7,8],
			'b':[0,12,13,14,15],
			'm':[13,17,18,19],
			'k':[13,25,26,27]}

def add_line(plt_canvas, coords, color, size):
	plt_canvas.plot(coords[:,1], coords[:,2], color=color, linewidth=size)

def plot_poses(batch, batch2=[], title='Poses', args='', image_dir=None):
	timesteps = len(batch[0])
	n = len(batch)
	n_total = len(batch) + len(batch2)
	f, axarr = plt.subplots(n_total, timesteps, sharex=True, sharey=True)

	for i in range(n):
		new_x = np.reshape(batch[i], (timesteps, -1, 3))
		for j in range(timesteps):
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[i, j], (new_x[j])[l], c, 1)

	for i in range(len(batch2)):
		new_x = np.reshape(batch2[i], (timesteps, -1, 3))
		for j in range(timesteps):
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[i+n, j], (new_x[j])[l], c, 2)


	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
	# plt.show()
	if image_dir == None:
		image_dir = '../new_out/'
	f.savefig(image_dir + title.lower().replace(' ', '_')+args+ strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png', dpi=192)
	plt.close(f)
