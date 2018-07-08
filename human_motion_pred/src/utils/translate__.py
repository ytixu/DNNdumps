import numpy as np

import data_utils__
import forward_kinematics__


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

  train_subject_ids = [1,6,7,8,9,11]
  test_subject_ids = [5]

  train_set, complete_train = data_utils__.load_data( data_dir, train_subject_ids, actions, one_hot )
  test_set,  complete_test  = data_utils__.load_data( data_dir, test_subject_ids,  actions, one_hot )

  # Compute normalization stats
  data_mean, data_std, dim_to_ignore, dim_to_use = data_utils__.normalization_stats(complete_train)

  # Normalize -- subtract mean, divide by stdev
  train_set = data_utils__.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
  test_set  = data_utils__.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
  print("done reading data.")

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def batch_expmap2euler(batch_data, config, labels):
  nframes = batch_data.shape[1]
  xyz = np.zeros((batch_data.shape[0], nframes, 96))

  srnn_pred_expmap = data_utils__.revert_output_format( batch_data,
            config['data_mean'], config['data_std'], config['dim_to_ignore'],
            config['actions'], labels)

  print 'srnn', srnn_pred_expmap

  # Put them together and revert the coordinate space
  expmap_all = forward_kinematics__.revert_coordinate_space( srnn_pred_expmap, np.eye(3), np.zeros(3) )

  # Compute 3d points for each frame
  parent, offset, rotInd, expmapInd = forward_kinematics__._some_variables()
  xyz = np.zeros((nframes, 96))
  for i in range( nframes ):
    xyz[i,:] = forward_kinematics__.fkl( expmap_all[i,:], parent, offset, rotInd, expmapInd )

  return xyz


  # for eulerchannels_pred in srnn_pred_expmap:
  #   # Convert from exponential map to Euler angles
  #   for j in np.arange( eulerchannels_pred.shape[0] ):
  #     for k in np.arange(3,97,3):
  #       eulerchannels_pred[j,k:k+3] = data_utils__.rotmat2euler(
  #         data_utils__.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

    # # The global translation (first 3 entries) and global rotation
    # # (next 3 entries) are also not considered in the error, so the_key
    # # are set to zero.
    # # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
    # gt_i=np.copy(srnn_gts_euler[action][i])
    # gt_i[:,0:6] = 0

    # # Now compute the l2 error. The following is numpy port of the error
    # # function provided by Ashesh Jain (in matlab), available at
    # # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
    # idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]

    # euc_error = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
    # euc_error = np.sum(euc_error, 1)
    # euc_error = np.sqrt( euc_error )
    # mean_errors[i,:] = euc_error
