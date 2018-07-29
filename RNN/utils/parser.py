import argparse
import time
import glob
import os.path
import numpy as np

TOGGLE_MODE = 'train'

def random_data_generator(timesteps, batch_size):
	# two gaussians
	for i in range(10000):
		data = np.random.random_sample((batch_size, timesteps, 2))*0.2-0.1
		a_ind = np.random.randint(0,batch_size,np.random.randint(batch_size/3))
		b_ind = list(set(range(batch_size)) - set(a_ind.tolist()))
		data[:,:,0] = data[:,:1,1]

		for j in range(1,timesteps):
			data[a_ind,j,1] = data[a_ind,j-1,1]-0.5/timesteps
			data[b_ind,j,1] = data[b_ind,j-1,1]+0.5/timesteps

		yield data[:,:,:1], data[:,:,1:]


def data_dimensions(input_dir, output_dir):
	for input_file in glob.glob(input_dir+'*'):
		name_id = os.path.basename(input_file)
		# 2D arrays
		input_data = np.load(input_file)
		output_data = np.load(output_dir+name_id)
		return input_data.shape[1], output_data.shape[1]

def get_one_hot_labels(files):
	labels = set(map(lambda x: os.path.basename(x).split('_')[0], files))
	return {l:i for i,l in enumerate(labels)}, len(labels)


def data_generator_random(input_dir, output_dir, timesteps, batch_size, n, label=False, ls=[], ld=0):
	data = {}
	total_i, x, y = None, None, None

	for i, input_file in enumerate(glob.glob(input_dir+'*')):
		name_id = os.path.basename(input_file)
		output_file = output_dir + name_id
		if label:
			name_label = name_id.split('_')[0]
			new_l = np.zeros(ld)
			new_l[ls[name_label]] = 1
			data[i] = (np.load(input_file), np.load(output_file), new_l)
		else:
			data[i] = (np.load(input_file), np.load(output_file))
		total_i = i

	def get_k(x):
		k = np.random.choice(data[x][0].shape[0]-timesteps, replace=False)
		if label:
			return (np.array([np.concatenate((data[x][0][k+t], data[x][2])) for t in range(timesteps)]),
				np.array([np.concatenate((data[x][1][k+t], data[x][2])) for t in range(timesteps)]))

		return (data[x][0][k:k+timesteps], data[x][1][k:k+timesteps])

	for i in range(n):
		idx = np.random.choice(total_i+1, batch_size)
		dd = [get_k(j) for j in idx]
		x, y = np.array([xx for xx,_ in dd]), np.array([yy for _,yy in dd])
		yield x, y

def data_generator(input_dir, output_dir, timesteps, batch_size, label=False, ls=[], ld=0, only_label=''):
	# files must be in different directories,
	# input and output filename must match for corresponding data
	# each file is one sequence

	# output one batch of <timesteps> data
	norm = 1
	if '_robot' in output_dir:
		norm = np.pi*2

	x, y, l = [], [], []
	name_label = ''
	batch_count = 0
	for input_file in glob.glob(input_dir+'*'):
		name_id = os.path.basename(input_file)
		if label:
			name_label = name_id.split('_')[0]
			new_l = np.zeros(ld)
			new_l[ls[name_label]] = 1
			if only_label != '' and name_label != only_label:
				continue

		# 2D arrays
		input_data = np.load(input_file)
		output_data = np.load(output_dir+name_id)

		batch = batch_size-batch_count
		for i in range(0, input_data.shape[0], batch):
			temp_x = input_data[i:i+batch+timesteps-1]
			count = len(temp_x)
			if count < timesteps:
				continue

			if label:
				new_x = np.array([[np.concatenate([temp_x[j+k], new_l], axis=0) for j in range(timesteps)] for k in range(count-timesteps+1)])
				new_y = np.array([[np.concatenate([output_data[i+j+k], new_l], axis=0) for j in range(timesteps)] for k in range(count-timesteps+1)])
			else:
				new_x = np.array([[temp_x[j+k] for j in range(timesteps)] for k in range(count-timesteps+1)])
				new_y = np.array([[output_data[i+j+k] for j in range(timesteps)] for k in range(count-timesteps+1)])

			if len(x) == 0:
				x = new_x
				y = new_y
			else:
				x = np.concatenate((x, new_x), axis=0)
				y = np.concatenate((y, new_y), axis=0)

			batch_count += len(new_x)
			if batch_count >= batch_size:
				batch_count = 0
				yield x, y/norm
				x, y = [], []
	if len(x) != 0:
		yield x, y/norm


def get_model_load_name(model_name):
	return '../models/%s_%d.hdf5'%(model_name, time.time())

def get_log_name(model_name):
	return '../new_out/%s_%d.csv'%(model_name, time.time())

def get_parse(model_name, labels=False):
	global TOGGLE_MODE
	ap = argparse.ArgumentParser()
	list_of_modes = ['train', 'sample', 'cont']
	ap.add_argument('-id', '--input_data', required=True, help='Input data directory')
	ap.add_argument('-od', '--output_data', required=True, help='Output data directory')
	ap.add_argument('-vid', '--validation_input_data', required=False, help='Validation input data directory')
	ap.add_argument('-vod', '--validation_output_data', required=False, help='Validation output data directory')
	ap.add_argument('-m', '--mode', required=False, help='Choose between training mode or sampling mode.', default='train', choices=list_of_modes)
	ap.add_argument('-ep', '--epochs', required=False, help='Number of epochs', default='1', type=int)
	ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default='16', type=int)
	ap.add_argument('-lp', '--load_path', required=False, help='Model path', default=get_model_load_name(model_name))
	ap.add_argument('-sp', '--save_path', required=False, help='Model save path', default=get_model_load_name(model_name))
	ap.add_argument('-t', '--timesteps', required=False, help='Timestep size', default='5', type=int)
	ap.add_argument('-p', '--periods', required=False, help='Number of iterations of the data', default='1', type=int)
	ap.add_argument('-ld', '--latent_dim', required=False, help='Embedding size', default='100', type=int)
	ap.add_argument('-o', '--option_dim', required=False, help='Number of options', default='2', type=int)
	ap.add_argument('-l', '--log_path', required=False, help='Log file for loss history', default=get_log_name(model_name))
        ap.add_argument('-only', '--only_label', required=False, help='Only load data with this label', default='')


	# ap.add_argument('-lr', '--learning_rate', required=False, help='Learning rate', default='5000', choices=list_of_modes)


	args = vars(ap.parse_args())
	if args['mode'] == 'cont':
		args['mode'] = 'sample'
		TOGGLE_MODE = 'sample'
		print TOGGLE_MODE
	if labels:
		ls, ld = get_one_hot_labels(glob.glob(args['input_data']+'*'))
		args['labels'] = {'purchases': 0, 'walking': 1, 'takingphoto': 2, 'eating': 3, 'sitting': 4, 'discussion': 5, 'walkingdog': 6, 'greeting': 7, 'walkingtogether': 8, 'phoning': 9, 'posing': 10, 'directions': 11, 'smoking': 12, 'waiting': 13, 'sittingdown': 14}
		args['label_dim'] = 15
		ls = args['labels']
		ld = args['label_dim']
	train_data = None
	if labels:
		if args['mode'] == TOGGLE_MODE:
			train_data = data_generator_random(args['input_data'], args['output_data'], args['timesteps'], 30000, 400, True, ls, ld)
		else:
			train_data = data_generator(args['input_data'], args['output_data'], args['timesteps'], 10000, True, ls, ld, only_label=args['only_label'])
	else:
		if args['mode'] == TOGGLE_MODE:
			train_data = data_generator_random(args['input_data'], args['output_data'], args['timesteps'], 30000, 400)
		else:
			train_data = data_generator(args['input_data'], args['output_data'], args['timesteps'], 10000, only_label=args['only_label'])

	validation_data = []
	if args['validation_input_data']:
		vd = None
		if labels:
			vd = data_generator_random(args['input_data'], args['output_data'], args['timesteps'], 30000, 400, True, ls, ld)
			#vd = data_generator(args['validation_input_data'], args['validation_input_data'], args['timesteps'], 10000000, True, ls, ld, only_label=args['only_label'])
		else:
			vd = data_generator(args['validation_input_data'], args['validation_input_data'], args['timesteps'], 10000000, only_label=args['only_label'])
		for v, _ in vd:
			validation_data = v
			break

	args['input_dim'], args['output_dim'] = data_dimensions(args['input_data'], args['output_data'])
	# data_iterator = random_data_generator(args['timesteps'], 5000)
	# args['input_dim'], args['output_dim'] = 1, 1
	return train_data, validation_data, args
