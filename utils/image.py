import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import gmtime, strftime

def add_point(plt_canvas, coords, color, size):
	l = len(coords)/2
	plt_canvas.scatter(x=coords[:l], y=coords[l:], c=color, s=size)
	# # plt_canvas.axis('off')
	plt_canvas.axes.get_xaxis().set_ticks([])
	plt_canvas.axes.get_yaxis().set_ticks([])
	plt_canvas.set_xlim(0, 1)
	plt_canvas.set_ylim(0, 1)

def plot_batch(batch, size, title='Poses'):
	plot_dim = int(np.sqrt(size))
	f, axarr = plt.subplots(plot_dim, plot_dim)

	for i, x in enumerate(batch):
		add_point(axarr[i/plot_dim, i%plot_dim], x, 'b', 10)
	
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
