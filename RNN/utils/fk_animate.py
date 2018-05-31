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


def animate_compare(true_seq, pred_seq, pred_name, baseline_seq, baseline_name, save_path):
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

	def init():
		return ob_true.get_lines() + ob_pred.get_lines() + ob_pred.get_lines()

	def animate(t):
		ob_true.update(true_seq[t])
		ob_baseline.update(baseline_seq[t])
		ob_pred.update(pred_seq[t])
		return init()


	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=true_seq.shape[0], interval=200, blit=True)
	anim.save(save_path+'animation.gif', writer='imagemagick', fps=60)



