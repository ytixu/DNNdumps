import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from time import gmtime, strftime, sleep

def add_point(plt_canvas, coords, color, size):
	l = len(coords)/2
	plt_canvas.scatter(x=coords[:l], y=coords[l:], c=color, s=size)
	# # plt_canvas.axis('off')
	plt_canvas.axes.get_xaxis().set_ticks([])
	plt_canvas.axes.get_yaxis().set_ticks([])
	plt_canvas.set_xlim(-1, 1)
	plt_canvas.set_ylim(-1, 1)

def animate(batch, dim=3):

	l = len(batch[0])/3
	fig = plt.figure()

	for pose in batch:
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlim(-1, 1)
		ax.set_ylim(-1, 1)
		ax.set_zlim(-1, 1)
		xs = pose[:l]
		ys = pose[l*2:]
		zs = pose[l:l*2]
		ax.scatter(xs, ys, zs, color='b')
		plt.pause(1)
		plt.clf()

def plot_batch(batch_true, batch_predict=[], size=0, title='Poses', dim=2):
	plot_dim = int(np.sqrt(size))
	if batch_predict.any():
		batch = np.concatenate([batch_true, batch_predict], axis=0)
	else:
		batch = batch_true
	sep_n = len(batch_true)
	def get_color(count):
		if count >= sep_n:
			return 'r'
		return 'b'

	if dim == 2:
		f, axarr = plt.subplots(plot_dim, plot_dim)
		for i, x in enumerate(batch):
			add_point(axarr[i/plot_dim, i%plot_dim], x, get_color(i), 10)
	else:
		f, axarr = plt.subplots(plot_dim, plot_dim*2)
		n = len(batch[0])/3
		for i, x in enumerate(batch):
			add_point(axarr[i/plot_dim, (i%plot_dim)*2], x[:n*2], get_color(i), 10)
			add_point(axarr[i/plot_dim, (i%plot_dim)*2+1], x[range(n*2, n*3) + range(n,n*2)], get_color(i), 10)
	
	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
		
	f.savefig('DNNdumps/out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close(f)
	# plt.show()
	# plt.pause(1)
	# plt.clf()

def plot_row(row, title='Interpolation'):
	plot_dim = len(row)
	f, axarr = plt.subplots(1, plot_dim)

	for i, x in enumerate(row):
		add_point(axarr[i%plot_dim], x, 'b', 10)
	
	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
		
	# f.savefig('DNNdumps/out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	# plt.close(f)
	plt.show()
	plt.pause(1)
	plt.clf()


def generate_and_plot(gan):
	fake_x = gan.sample()
	plot_batch(fake_x, gan.batch_size, 'Generated poses')
