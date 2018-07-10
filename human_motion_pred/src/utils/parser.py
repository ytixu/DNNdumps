import argparse
import time
import os.path
import numpy as np
import json

import translate__
import data_utils__

MODEL_DIR = '../models/'
OUTPUT_DIR = '../new_out/'

def get_save_name(name):
  return MODEL_DIR+'%s_%d.hdf5'%(name, time.time())

def get_log_name(name):
  return OUTPUT_DIR+'%s_%d.csv'%(name, time.time())

def get_parse(model_name, labels=False, create_params=False):
  ap = argparse.ArgumentParser()
  list_of_modes = ['train', 'sample', 'cont']
  ap.add_argument('-dd', '--data_dir', required=False, help='Data directory', default='../../data/h3.6/raw/h3.6m/dataset')
  ap.add_argument('-dp', '--data_param', required=False, help='Data parameter file path (the mean, std and used dims)', default='../../data/h3.6/raw/h3.6m/params.json')

  ap.add_argument('-load', '--load_path', required=False, help='Model path', default='')
  ap.add_argument('-save', '--save_path', required=False, help='Model save path', default='')
  ap.add_argument('-log', '--log_path', required=False, help='Log file for loss history', default='')

  ap.add_argument('-m', '--mode', required=False, help='Choose between training mode or sampling mode.', default='train', choices=list_of_modes)
  ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default='16', type=int)
  # ap.add_argument('-ep', '--epochs', required=False, help='Number of epochs', default='1', type=int)
  ap.add_argument('-p', '--periods', required=False, help='Number of iterations of the data', default='100000', type=int)
  ap.add_argument('-rn', '--rand_n', required=False, help='Number of data per iteration', default='10000', type=int)
  ap.add_argument('-t', '--timesteps', required=False, help='Timestep size', default='75', type=int)
  ap.add_argument('-ls', '--hierarchies', required=False, nargs='+', help='The sequence lengths to train on. all means all the lengths', default='all')
  ap.add_argument('-ld', '--latent_dim', required=False, help='Embedding size', default='1024', type=int)
  ap.add_argument('-lr', '--learning_rate', required=False, help='Learning rate.', default='0.005')
  ap.add_argument('-te', '--test_every', required=False, help='How often to compute error on the test set.', default='10')

  ap.add_argument('-a', '--actions', required=False, help='The action to train on. all means all the actions, all_periodic means walking, eating and smoking', default='all')

  # Learning
  # tf.app.flags.DEFINE_float("learning_rate", .005, "Learning rate.")
  # tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate is multiplied by this much. 1 means no decay.")
  # tf.app.flags.DEFINE_integer("learning_rate_step", 10000, "Every this many steps, do decay.")
  # tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
  # tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
  # tf.app.flags.DEFINE_integer("iterations", int(1e5), "Iterations to train for.")
  # # Architecture
  # tf.app.flags.DEFINE_string("architecture", "tied", "Seq2seq architecture to use: [basic, tied].")
  # tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
  # tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
  # tf.app.flags.DEFINE_integer("seq_length_in", 50, "Number of frames to feed into the encoder. 25 fps")
  # tf.app.flags.DEFINE_integer("seq_length_out", 10, "Number of frames that the decoder has to predict. 25fps")
  # tf.app.flags.DEFINE_boolean("omit_one_hot", False, "Whether to remove one-hot encoding from the data")
  # tf.app.flags.DEFINE_boolean("residual_velocities", False, "Add a residual connection that effectively models velocities")
  # # Directories
  # tf.app.flags.DEFINE_string("data_dir", os.path.normpath("./data/h3.6m/dataset"), "Data directory")
  # tf.app.flags.DEFINE_string("train_dir", os.path.normpath("./experiments/"), "Training directory.")

  # tf.app.flags.DEFINE_string("action","all", "The action to train on. all means all the actions, all_periodic means walking, eating and smoking")
  # tf.app.flags.DEFINE_string("loss_to_use","sampling_based", "The type of loss to use, supervised or sampling_based")

  # tf.app.flags.DEFINE_integer("test_every", 1000, "How often to compute error on the test set.")
  # tf.app.flags.DEFINE_integer("save_every", 1000, "How often to compute error on the test set.")
  # tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
  # tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
  # tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")


  args = vars(ap.parse_args())
  model_signature = '%s_t%d_l%d' % (model_name, args['timesteps'], args['latent_dim'])

  # set up log and save paths
  if args['save_path'] == '':
    args['save_path'] = get_save_name(model_signature)
  if args['log_path'] == '':
    args['log_path'] = get_log_name(model_signature)
  args['model_signature'] = model_signature

  actions = translate__.define_actions(args['actions'])
  number_of_actions = len(actions)
  args['actions'] = actions

  if create_params:
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = translate__.read_all_data(
      actions, args['timesteps'], args['data_dir'], labels)

    args['data_mean'] = data_mean
    args['data_std'] = data_std
    args['dim_to_ignore'] = dim_to_ignore
    args['dim_to_use'] = dim_to_use
    args['data_dim'] = len(dim_to_use)

  else:
    with open(args['data_param'], 'rb') as param_file:
      args.update(json.load(param_file))
      args['data_std'] = np.array(args['data_std'])
      args['data_mean'] = np.array(args['data_mean'])

    args['data_dim'] = len(args['dim_to_use'])

    train_set = translate__.train_data_randGen( args, labels )
    test_set = translate__.get_test_data( args, labels )

  if labels:
    args['label_dim'] = number_of_actions

  return train_set, test_set, args

if __name__ == '__main__':
  # test
  train_set, test_set, args = get_parse('TEST', create_params=True)

  print args
  print train_set.values()[0].shape
  print test_set.values()[0].shape
  print args['data_mean']
  print args['data_std']
  print args['dim_to_ignore']
  print args['dim_to_use']

  # create data parameter file
  import math
  with open(args['data_param'], 'wb') as param_file:
    json.dump({
      'data_mean':args['data_mean'].tolist(),
      'data_std':args['data_std'].tolist(),
      'dim_to_ignore':args['dim_to_ignore'],
      'dim_to_use':args['dim_to_use'],
      'data_max':math.ceil(max([np.max(v) for v in train_set.values()] + [np.max(v) for v in test_set.values()])),
      'data_min':math.floor(min([np.min(v) for v in train_set.values()] + [np.min(v) for v in test_set.values()]))
    }, param_file)
