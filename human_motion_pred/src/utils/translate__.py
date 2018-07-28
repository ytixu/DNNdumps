import numpy as np

import data_utils__
import forward_kinematics__

TRAIN_SUBJECT_ID = [1,6,7,8,9,11]
TEST_SUBJECT_ID = [5]

def define_actions( action ):
  """
  Define the list of actions we are using.
  Args
    action: String with the passed action. Could be "all"
  Returns
    actions: List of strings of actions
  Raises
    ValueError if the action is not included in H3.6M
  """

  actions = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]

  if action in actions:
    return [action]

  if action == "all":
    return actions

  if action == "all_srnn":
    return ["walking", "eating", "smoking", "discussion"]

  raise( ValueError, "Unrecognized action: %d" % action )

def read_all_data( actions, seq_length, data_dir, one_hot ):
  """
  Loads data for training/testing and normalizes it.
  Args
    actions: list of strings (actions) to load
    seq_length: number of frames
    data_dir: directory to load the data from
    one_hot: whether to use one-hot encoding per action
  Returns
    train_set: dictionary with normalized training data
    test_set: dictionary with test data
    data_mean: d-long vector with the mean of the training data
    data_std: d-long vector with the standard dev of the training data
    dim_to_ignore: dimensions that are not used becaused stdev is too small
    dim_to_use: dimensions that we are actually using in the model
  """

  # === Read training data ===
  print "Reading training data (seq_len: %d)" % (seq_length)

  train_set, complete_train = data_utils__.load_data( data_dir, TRAIN_SUBJECT_ID, actions, one_hot )
  test_set,  complete_test  = data_utils__.load_data( data_dir, TEST_SUBJECT_ID,  actions, one_hot )

  # Compute normalization stats
  data_mean, data_std, dim_to_ignore, dim_to_use = data_utils__.normalization_stats(complete_train)

  # Normalize -- subtract mean, divide by stdev
  train_set = data_utils__.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
  test_set  = data_utils__.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
  print("done reading data.")

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

def train_data_for_testing( config, one_hot ):
  for key, train_set in data_utils__.load_data( config['data_dir'], TRAIN_SUBJECT_ID, config['actions'], one_hot, generator=True ):
    #train_set = np.array([train_set[i:i+config['timesteps']] for i in range(train_set.shape[0]-config['timesteps'])])
    train_set = np.array([train_set])

    train_set = data_utils__.normalize_data( train_set, config['data_mean'],
      config['data_std'], config['dim_to_use'], config['actions'], one_hot,
      config['data_max'], config['data_min'] )

    yield key, train_set[0]

def train_data_randGen( config, one_hot ):
  for train_set in data_utils__.load_rand_data( config['data_dir'],
      TRAIN_SUBJECT_ID, config['actions'], one_hot, config['timesteps'],
      config['rand_n'], config['periods']):

    train_set = data_utils__.normalize_data( train_set, config['data_mean'],
      config['data_std'], config['dim_to_use'], config['actions'], one_hot,
      config['data_max'], config['data_min'] )

    yield train_set

def get_test_data( config, one_hot ):
  def x():
    expmap_gt, expmap_pred_gt = data_utils__.get_test_data( config['data_dir'],
      TEST_SUBJECT_ID, config['actions'], one_hot )

    expmap_gt = data_utils__.normalize_data( expmap_gt, config['data_mean'],
      config['data_std'], config['dim_to_use'], config['actions'], one_hot,
      config['data_max'], config['data_min'] )

    expmap_pred_gt = data_utils__.normalize_data( expmap_pred_gt,
      config['data_mean'], config['data_std'], config['dim_to_use'],
      config['actions'], one_hot, config['data_max'], config['data_min'] )
    return expmap_gt, expmap_pred_gt

  return x

def batch_convert_expmap(batch_data, model):
  '''
    Unormalize a batch of exponential map for later conversions
    (to euler angles or to euclidean space)
  '''
  for i in np.arange( batch_data.shape[0] ):
      yield data_utils__.unNormalizeData(batch_data[i,:,:], model.data_mean,
        model.data_std, model.dim_to_ignore, model.labels, model.has_labels,
        model.data_max, model.data_min )
      #yield batch_data[i,:,:]

def batch_expmap2euler(batch_data, model, normalized=True):
  '''
    Convert a batch of exponential map to euler angle
  '''
  srnn_euler = [None]*batch_data.shape[0]

  if normalized:
    batch_data = batch_convert_expmap(batch_data, model)

  for i, denormed in enumerate(batch_data):
    for j in np.arange( denormed.shape[0] ):
      for k in np.arange(3,97,3):
        denormed[j,k:k+3] = data_utils__.rotmat2euler( data_utils__.expmap2rotmat( denormed[j,k:k+3] ))
    srnn_euler[i] = denormed

  return srnn_euler

def batch_expmap2xyz(batch_data, model, normalized=True):
  '''
    Convert a batch of exponential map to euclidean space using FK
  '''
  data_copy = np.copy(batch_data)

  nSamples, nframes, _ = data_copy.shape
  xyz = np.zeros((nSamples, nframes, 96))

  if normalized:
    batch_data = batch_convert_expmap(data_copy, model)

  for i, denormed in enumerate(batch_data):
    # Put them together and revert the coordinate space
    deformed = forward_kinematics__.revert_coordinate_space( denormed, np.eye(3), np.zeros(3) )

    # Compute 3d points for each frame
    parent, offset, rotInd, expmapInd = forward_kinematics__._some_variables()
    for j in range( nframes ):
      xyz[i,j,:] = forward_kinematics__.fkl( deformed[j,:], parent, offset, rotInd, expmapInd )

  print 'xyz', xyz.shape
  return xyz

def euler_diff(batch_expmap1, batch_expmap2, model, normalized=[True, True]):
  '''
    Get the mean euler angle difference between two batches
  '''
  copy1 = np.copy(batch_expmap1)
  copy2 = np.copy(batch_expmap2)

  batch_euler1 = batch_expmap2euler(copy1, model, normalized[0])
  batch_euler2 = batch_expmap2euler(copy2, model, normalized[1])

  # Compute and save the errors here
  mean_errors = np.zeros( (len(batch_expmap1), batch_expmap1[0].shape[0]) )

  for i, srnn_euler in enumerate(batch_euler1):
    # The global translation (first 3 entries) and global rotation
    # (next 3 entries) are also not considered in the error, so the_key
    # are set to zero.
    # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
    srnn_euler[:,0:6] = 0

    # Now compute the l2 error. The following is numpy port of the error
    # function provided by Ashesh Jain (in matlab), available at
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
    idx_to_use = np.where( np.std( srnn_euler, 0 ) > 1e-4 )[0]

    euc_error = np.power( srnn_euler[:,idx_to_use] - batch_euler2[i][:,idx_to_use], 2)
    euc_error = np.sum(euc_error, 1)
    euc_error = np.sqrt( euc_error )
    mean_errors[i,:] = euc_error

    mean_mean_errors = np.mean( mean_errors, 0 )

  return mean_mean_errors, mean_errors
