import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from time import gmtime, strftime


# def plot_time_series(x, title):
# 	plt.scatter(x, range(len(x)))
# 	plt.suptitle(title)
# 	plt.show()

def add_point(plt_canvas, coords, color, size):
	l = len(coords)/2
	plt_canvas.scatter(x=coords[:,0], y=coords[:,1], c=color, s=size)
	# # plt_canvas.axis('off')
	# plt_canvas.axes.get_xaxis().set_ticks([])
	# plt_canvas.axes.get_yaxis().set_ticks([])
	# plt_canvas.set_xlim(min_lim, max_lim)
	# plt_canvas.set_ylim(min_lim, max_lim)


def plot_batch_3D(batch_true, batch_predict, title='Poses (prediction in blue)'):
	size = len(batch_true[0])
	f, axarr = plt.subplots(4, size, sharex=True, sharey=True)
	n = len(batch_true[0][0])/3
	for i, x in enumerate(batch_true[0]):
		x_reshaped = x.reshape((n, 3))
		add_point(axarr[0, i], x_reshaped[:,:2], 'r', 10)
		add_point(axarr[2, i], x_reshaped[:,1:], 'r', 10)
		x_reshaped = batch_predict[0][i].reshape((n, 3))
		add_point(axarr[1, i], x_reshaped[:,:2], 'b', 10)
		add_point(axarr[3, i], x_reshaped[:,1:], 'b', 10)

	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
		
	f.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close(f)

def plot_batch_1D(batch_true, batch_predict, title='Poses (prediction in blue)'):
	size = len(batch_true[0])
	n = len(batch_true[0][0])
	f, axarr = plt.subplots(2, size, sharex=True, sharey=True)
	for i, x in enumerate(batch_true[0]):
		new_x = np.zeros((n, 2))
		new_x[:,0] = range(n)
		new_x[:,1] = x
		add_point(axarr[0, i], new_x, 'r', 10,)
		new_x[:,1] = batch_predict[0][i]
		add_point(axarr[1, i], new_x, 'b', 10)

	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
		
	f.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close(f)
