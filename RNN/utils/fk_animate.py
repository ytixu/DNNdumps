import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import images.viz as viz
import metrics

def animate_random(model, start_seq, embedding, mean=0.38919052, std=0.1443201):
	h = model.timesteps

	fig = plt.figure()
	ax = plt.gca(projection='3d')
	ob = viz.Ax3DPose(ax)

	pose = start_seq
	enc = metrics.__get_latent_reps(model.encoder, np.array([pose]), model.MODEL_CODE, h-1)


	while True:

		for t in range(model.timesteps):
			ob.update(pose[t])
			plt.show(block=False)
			fig.canvas.draw()
			plt.pause(0.01)

		# enc = enc + np.random.normal(mean, std)
		# print enc.shape
		pose = metrics.__get_consecutive(pose, model, h/2)
		# poses = metrics.__get_decoded_reps(model.decoder, enc, model.MODEL_CODE)


def animate_motion(seq, name, save_path):
	if type(name) == type('str'):
		seq = [seq]
		name = [name]

	fig = plt.figure()
	n = len(seq)
	axs = [None]*n
	obs = [None]*n
	for i in range(n):
		axs[i] = fig.add_subplot(1, len(seq), i+1, projection='3d')
		axs[i].set_title(name[i])
		obs[i] = viz.Ax3DPose(axs[i])

	n_t = seq[0].shape[0]

	def init():
		if n == 1:
			return obs[0].get_lines()
		return reduce(lambda acc, x: acc + x, [obs[i].get_lines() for i in range(n)])

	def animate(t):
		for i in range(n):
			obs[i].update(seq[i][t])
		return init()


	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_t, interval=400, blit=True)
	filename = save_path+'-'.join(name)+'.gif'
	anim.save(filename, writer='imagemagick', fps=60)


def animate_compare(start_seq, true_seq, pred_seq, pred_name, baseline_seq, baseline_name, save_path):
	fig = plt.figure()
	ax_true = fig.add_subplot(1, 3, 1, projection='3d')
	ax_baseline = fig.add_subplot(1, 3, 2, projection='3d')
	ax_pred = fig.add_subplot(1, 3, 3, projection='3d')

	ax_true.set_title('Ground Truth')
	ax_baseline.set_title(baseline_name)
	ax_pred.set_title(pred_name)

	ob_true = viz.Ax3DPose(ax_true)
	ob_baseline = viz.Ax3DPose(ax_baseline)
	ob_pred = viz.Ax3DPose(ax_pred)

	n_start = start_seq.shape[0]

	def init():
		return ob_true.get_lines() + ob_pred.get_lines() + ob_pred.get_lines()

	def animate(t):
		if t >= n_start:
			ob_true.update(true_seq[t-n_start])
			ob_baseline.update(baseline_seq[t-n_start])
			ob_pred.update(pred_seq[t-n_start])
		else:
			ob_true.update(start_seq[t], '#000000')
			ob_baseline.update(start_seq[t], '#000000')
			ob_pred.update(start_seq[t], '#000000')
		return init()


	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=true_seq.shape[0]+n_start, interval=400, blit=True)
	filename = save_path +'.gif'
	anim.save(filename, writer='imagemagick', fps=60)
