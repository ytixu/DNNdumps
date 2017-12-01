import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def random_3D_data(n=50, b=16, j=6):
	x = np.zeros((n*b, j*3))
	for i in range(j*3):
		coord = np.random.randint(1,9)
		x[:,i] = np.random.normal(coord, 0.1, n*b)
		print i, coord

	return x


def generate_and_plot_3D(gan):
	noise_x = np.random.uniform(-1.0, 1.0, size=[gan.batch_size, gan.latent_size])
	fake_x = gan.generator().predict(noise_x)
	
	for x in fake_x:
		xs = [x[i]*gan.norm_max for i in range(0, len(x), 3)]
		ys = [x[i]*gan.norm_max for i in range(1, len(x), 3)]
		zs = [x[i]*gan.norm_max for i in range(2, len(x), 3)]

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(xs, ys, zs)
		plt.show()