import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def add_point(coords, color, size):
	coords = np.round(coords).astype(int)
	n = len(coords)/2
	plt.scatter(x=coords[:n], y=coords[n:], c=color, s=size)

def generate_and_plot(gan):
	noise_x = np.random.uniform(-1.0, 1.0, size=[gan.batch_size, gan.latent_size])
	fake_x = gan.generator().predict(noise_x)
	
	n = len(fake_x[0])/2
	for x in fake_x:
		add_point(x, 'b', 40)
		# plt.show()
		plt.pause(0.05)
		plt.clf()