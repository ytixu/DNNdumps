import numpy as np
from glob import glob
from tqdm import tqdm
import json

directory = '../../h3.6/full/'

def get_data(files):
	data = []
	for i,f in enumerate(tqdm(files)):
		if len(data) == 0:
			data = np.load(f)
		else:
			data = np.concatenate([np.load(f), data], axis = 0)
	print data.shape
	return data



if __name__ == '__main__':
	for parameterization in ['_cart', '_euler']:
		print parameterization
		files = glob(directory+'*'+parameterization+'/*.npy')
		data = data = get_data(files)

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
