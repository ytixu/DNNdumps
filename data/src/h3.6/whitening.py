import numpy as np
from glob import glob



if __name__ == '__main__':
	for parameterization in ['_cart', '_euler']:
		directory = '../../h3.6/full/'
		files = glob(directory+'*'+parameterization'/*.npy')

		data = [None]*len(files)

		for i,f in enumerate(files):
			data[i] = np.load(f)

		print data.shape

		data_mean = np.mean(data, axis=0)
		data_std = np.std(data, axis=0)

		dimensions_to_ignore = list(np.where(data_std < 1e-4)[0])
		dimensions_to_use = list(np.where(data_std >= 1e-4)[0])

		print len(dimensions_to_use), dimensions_to_ignore

		data_std[dimensions_to_ignore] = 1.0

		with open(directory+'stats'+parameterization+'.json', 'wb') as param_file:
			json.dump({
				'data_mean':data_mean.tolist(),
				'data_std':data_std.tolist(),
				'dim_to_ignore':dimensions_to_ignore,
				'dim_to_use':dimensions_to_use,
			}, param_file)