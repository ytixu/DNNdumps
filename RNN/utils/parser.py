import argparse
import time
import glob
import os.path
import numpy as np

def data_dimensions(input_dir, output_dir):
	for input_file in glob.glob(input_dir+'*'):
		name_id = os.path.basename(input_file)
		# 2D arrays
		input_data = np.load(input_file)
		output_data = np.load(output_dir+name_id)
		return input_data.shape[1], output_data.shape[1]


def data_generator(input_dir, output_dir, timesteps, batch_size):
	# files must be in different directories, 
	# input and output filename must match for corresponding data
	# each file is one sequence

	# output one batch of <timesteps> data

	x, y = [], []
	batch_count = 0
	for input_file in glob.glob(input_dir+'*'):
		name_id = os.path.basename(input_file)

		# 2D arrays
		input_data = np.load(input_file)
		output_data = np.load(output_dir+name_id)

		batch = batch_size-batch_count
		for i in range(0, input_data.shape[0], batch):
			temp_x = input_data[i:i+batch+timesteps-1]
			count = len(temp_x)
			if count < timesteps:
				continue
			new_x = np.array([[temp_x[j+k] for j in range(timesteps)] for k in range(count-timesteps+1)])
			new_y = np.array([[output_data[i+j+k] for j in range(timesteps)] for k in range(count-timesteps+1)])
			if len(x) == 0:
				x = new_x
				y = new_y
			else:
				x = np.concatenate((x, new_x), axis=0)
				y = np.concatenate((y, new_y), axis=0)

			batch_count += len(new_x)
			if batch_count == batch_size:
				batch_count = 0
				yield x, y



def get_parse(model_name):
	ap = argparse.ArgumentParser()
	list_of_modes = ['train', 'sample']
	ap.add_argument('-id', '--input_data', required=True, help='Input data directory')
	ap.add_argument('-od', '--output_data', required=True, help='Output data directory')
	ap.add_argument('-m', '--mode', required=False, help='Choose between training mode or sampling mode.', default='train', choices=list_of_modes)
	ap.add_argument('-ep', '--epochs', required=False, help='Number of epochs', default='10', type=int)
	ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default='16', type=int)
	ap.add_argument('-lp', '--load_path', required=False, help='Model path', default='DNNdumps/models/%s_%d.hdf5'%(model_name, time.time()))
	ap.add_argument('-t', '--timesteps', required=False, help='Timestep size', default='5', type=int)
	# ap.add_argument('-lr', '--learning_rate', required=False, help='Learning rate', default='5000', choices=list_of_modes)

	args = vars(ap.parse_args())
	data_iterator = data_generator(args['input_data'], args['output_data'], args['timesteps'], 1000)
	args['input_dim'], args['output_dim'] = data_dimensions(args['input_data'], args['output_data'])

	return data_iterator, args
