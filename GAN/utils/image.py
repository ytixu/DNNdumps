import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import gmtime, strftime

def add_point(plt_canvas, coords, color, size):
	plt_canvas.scatter(x=coords[:,0], y=coords[:,1], c=color, s=size)
	# # plt_canvas.axis('off')
	plt_canvas.axes.get_xaxis().set_ticks([])
	plt_canvas.axes.get_yaxis().set_ticks([])

def plot_batch(batch, size, title='Human poses'):
	plot_dim = int(np.sqrt(size))
	f, axarr = plt.subplots(plot_dim, plot_dim)

	for i, x in enumerate(batch):
		add_point(axarr[i/plot_dim, i%plot_dim], x, 'b', 10)
	
	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
		
	f.savefig('out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close(f)
	# plt.show()
	# plt.pause(1)
	# plt.clf()

def generate_and_plot(gan):
	fake_x = gan.sample()
	plot_batch(fake_x, gan.batch_size, 'Generated poses')
