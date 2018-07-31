import numpy as np
import matplotlib.pyplot as plt
import random
# from mpl_toolkits.mplot3d import Axes3D
from time import gmtime, strftime


# def plot_time_series(x, title):
# 	plt.scatter(x, range(len(x)))
# 	plt.suptitle(title)
# 	plt.show()

def get_color(n):
	c = [None]*n
	for i in range(n):
		r = lambda: random.randint(0,255)
		c[i] = '#%02X%02X%02X' % (r(),r(),r())
	return c

def add_line(plt_canvas, coords, color, size):
	plt_canvas.plot(coords[:,1], coords[:,2], color=color, linewidth=size)

def add_point(plt_canvas, coords, color, size):
	l = len(coords)/2
	plt_canvas.scatter(x=coords[:,0], y=coords[:,2], c=color, s=size)
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

# def plot_batch_2D(batch_true, batch_predict, title='Poses (prediction in blue)'):
# 	size = len(batch_true[0])
# 	f, axarr = plt.subplots(2, size, sharex=True, sharey=True)
# 	for i, p in enumerate(batch_true[0]):
# 		new_p = np.reshape(p, (-1,2))
# 		add_point(axarr[0, i], new_p, 'r', 10)
# 		new_p = np.reshape(batch_predict[0][i], (-1,2))
# 		add_point(axarr[1, i], new_p, 'b', 10)

# 	f.subplots_adjust(hspace=0.1)
# 	plt.suptitle(title)
		
# 	f.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
# 	plt.close(f)


def plot_batch_2D(batch_complete, batch_true, batch_predict, best_opt, p_best_opt, title='Poses (Green, Cyan: predicted best option)'):
	size = len(batch_true[0])
	f, axarr = plt.subplots(len(batch_predict[0])+1, size, sharex=True, sharey=True)
	for i, p in enumerate(batch_true[0]):
		new_p = np.reshape(p, (-1,2))
		pose = np.reshape(batch_complete[0][i], (-1, 2))
		add_point(axarr[0, i], new_p, 'r', 10)
		add_point(axarr[0, i], pose, 'y', 10)

		for j in range(len(batch_predict[0])):
			new_p = np.reshape(batch_predict[0][j][i], (-1,2))
			c = 'b'
			if j == best_opt[0]:
				if best_opt[0] == p_best_opt[0]:
					c = 'g'
				else:
					c = 'm'
			elif j == p_best_opt[0]:
				c = 'c'
			add_point(axarr[j+1, i], new_p, c, 10)
			add_point(axarr[j+1, i], pose, 'y', 10)

	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
		
	f.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close(f)

def plot_batch_1D(batch_true, batch_predict, title='Poses (prediction in blue)'):
	size = len(batch_true[0])
	n = len(batch_true[0][0])
	f, axarr = plt.subplots(2, size, sharex=True, sharey=True)
	new_x = np.zeros((n, 2))
	new_x[:,0] = range(n)
	for i, x in enumerate(batch_true[0]):
		new_x[:,1] = x
		add_point(axarr[0, i], new_x, 'r', 10)
		new_x[:,1] = batch_predict[0][i]
		add_point(axarr[1, i], new_x, 'b', 10)

	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
		
	f.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close(f)

def plot_graph(batch_x, batch_true, batch_predict, best_opt=[], p_best_opt=[], title='Poses (Green, Cyan: predicted best option)'):
	t = np.arange(len(batch_true[0]))
	for i, p in enumerate(batch_true):
		plt.plot(t+batch_x[i].flatten(), p.flatten(), c='r')
	for i, p in enumerate(batch_predict):
		for j in range(len(p)):
			c = 'b'
			if len(best_opt) > 0:
				if j == best_opt[i]:
					if best_opt[i] == p_best_opt[i]:
						c = 'g'
					else:
						c = 'm'
				elif j == p_best_opt[i]:
					c = 'c'

			plt.plot(t+batch_x[i].flatten(), p[j].flatten(), c=c)
	plt.suptitle(title)
	plt.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close()

def plot_options(ref, opts, title='Options'):
	colors = get_color(len(opts[0])+1)
	size = len(ref[0])
	n = len(ref[0][0])
	new_x = np.zeros((n*size, 2))
	new_x[:,0] = np.concatenate([np.arange(size)*1.0/(n+1)+i for i in range(n)])
	print new_x[:,0]
	new_x[:,1] = ref[0].T.flatten()
	add_point(plt, new_x, colors[0], 10)
	for j, opt in enumerate(opts[0]):
		new_x[:,1] = opt.T.flatten()
		add_point(plt, new_x, colors[j+1], 10)

	plt.suptitle(title)
	plt.show()
	# plt.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	# plt.close(f)

# M_POSE_LINES = {'r':[1, 0],
# 				'g':[0, 2, 3, 4, 5],
# 				'b':[0, 6, 7, 8, 9],
# 				'm':[0, 10, 11, 12],
# 				'k':[0, 13, 14, 15]}

M_POSE_LINES = {'r':[0,1,2,3],
				'g':[0,4,5,6],
				'b':[0,7,8,9,10],
				'm':[8,11,12,13],
				'k':[9,14,15,16]}


def plot_hierarchies(batch_true, batch_predict, title='Reference is the first line'):
	timesteps = len(batch_true[0])
	hierarchies = len(batch_predict[0])/timesteps
	n = 5
	skip_t = timesteps/n
	skip_h = hierarchies/n
	f, axarr = plt.subplots(n+1, n, sharex=True, sharey=True)
	for i in range(n):
		new_x = np.reshape(batch_true[0][i*skip_t], (-1, 3))
		for c, l in M_POSE_LINES.iteritems():
			add_line(axarr[0, i], new_x[l], c, 3)
		
		for j in range(n):
			new_x = np.reshape(batch_predict[0][j*skip_h*hierarchies+i*skip_t], (-1, 3))
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[j+1, i], new_x[l], c, 3)

	f.subplots_adjust(hspace=0.1)
	plt.suptitle(title)
	# plt.show()
	f.savefig('../out/' + title.lower().replace(' ', '_') + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close(f)

def plot_options_hierarchies(batch_true, batch_predict, title='Reference is the first line'):
	print batch_predict.shape
	batch, hierarchies, options, _ = batch_predict.shape
	_, timesteps, _ = batch_true.shape
	batch_predict = np.reshape(batch_predict, (batch, hierarchies, options, timesteps,-1))
	n = 5
	skip_t = timesteps/n
	timestamp = strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime())
	for h in range(0, hierarchies, 2):
		f, axarr = plt.subplots(options+1, n, sharex=True, sharey=True)
		for i in range(n):
			new_x = np.reshape(batch_true[0][i*skip_t], (-1, 3))
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[0, i], new_x[l], c, 3)
			
			for j in range(options):
				new_x = np.reshape(batch_predict[0,h,j,i*skip_t], (-1, 3))
				for c, l in M_POSE_LINES.iteritems():
					add_line(axarr[j+1, i], new_x[l], c, 3)

		f.subplots_adjust(hspace=0.1)
		plt.suptitle(title)
		# plt.show()
		f.savefig('../out/' + title.lower().replace(' ', '_') + timestamp + '--%d'%(h) + '.png') 
		plt.close(f)


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


def plot_poses_euler(batch, batch2=[], title='Poses', args='', image_dir=None):
	timesteps = len(batch[0])
	n = len(batch)
	n_total = len(batch) + len(batch2)
	skip_n = timesteps
	skip_t = timesteps/skip_n
	f, axarr = plt.subplots(n_total, skip_n, sharex=True, sharey=True)

	pose_template = np.zeros((timesteps, batch.shape[-1]+3))

	for i in range(n):
		pose_template[:,3:] = batch[i]
		new_x = np.reshape(pose_template, (timesteps, -1, 3))
		for j in range(skip_n):
			for c, l in M_POSE_LINES.iteritems():
				add_line(axarr[i, j], (new_x[j*skip_t])[l], c, 1)

	for i in range(len(batch2)):
		pose_template[:,3:] = batch2[i]
		new_x = np.reshape(pose_template, (timesteps, -1, 3))
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
