import argparse
import time
import glob
import os.path
import numpy as np

def get_model_load_name(model_name):
	return '../models/%s_%d.hdf5'%(model_name, time.time())

def get_log_name(model_name):
	return '../models/%s_%d.log'%(model_name, time.time())

def data_generator():
	pass

def get_parse(model_name, labels=False):
	ap = argparse.ArgumentParser()
	list_of_modes = ['train', 'sample']
	ap.add_argument('-id', '--input_data', required=True, help='Input data directory')
	ap.add_argument('-od', '--output_data', required=True, help='Output data directory')
	ap.add_argument('-m', '--mode', required=False, help='Choose between training mode or sampling mode.', default='train', choices=list_of_modes)
	ap.add_argument('-ep', '--epochs', required=False, help='Number of epochs', default='1', type=int)
	ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default='16', type=int)
	ap.add_argument('-lp', '--load_path', required=False, help='Model path', default=get_model_load_name(model_name))
	ap.add_argument('-sp', '--save_path', required=False, help='Model save path', default=get_model_load_name(model_name))
	ap.add_argument('-p', '--periods', required=False, help='Number of iterations of the data', default='1', type=int)

	train_data = data_generator(args['input_data'], args['output_data'])
	args['input_dim'], args['output_dim'] = data_dimensions(args['input_data'], args['output_data'])

	return train_data, args
