import numpy as np

import data_utils, foward_kinematics
from six.moves import xrange  # pylint: disable=redefined-builtin

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


seq_length_in = 1
seq_length_out = 1
batch_size = 16
actions = define_actions('all_srnn')
number_of_actions = len(actions)
one_hot = False
HUMAN_SIZE = 54
omit_one_hot = False
input_size = HUMAN_SIZE + number_of_actions if one_hot else HUMAN_SIZE


def read_all_data( actions, seq_length_in, seq_length_out, data_dir, one_hot ):
  """
  Loads data for training/testing and normalizes it.

  Args
    actions: list of strings (actions) to load
    seq_length_in: number of frames to use in the burn-in sequence
    seq_length_out: number of frames to use in the output sequence
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
  print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
           seq_length_in, seq_length_out))

  train_subject_ids = [1]#[1,6,7,8,9,11]
  test_subject_ids = [1,6,5,7,8,9,11]

  train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions, one_hot )
  test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions, one_hot )

  # Compute normalization stats
  data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

  # Normalize -- subtract mean, divide by stdev
  train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
  test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
  print("done reading data.")

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

def find_indices_srnn(data, action ):
  """
  Find the same action indices as in SRNN.
  See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
  """

  # Used a fixed dummy seed, following
  # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
  SEED = 1234567890
  rng = np.random.RandomState( SEED )

  subject = 5
  subaction1 = 1
  subaction2 = 2

  T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
  T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
  prefix, suffix = 50, 100

  idx = []
  idx.append( rng.randint( 16,T1-prefix-suffix ))
  idx.append( rng.randint( 16,T2-prefix-suffix ))
  idx.append( rng.randint( 16,T1-prefix-suffix ))
  idx.append( rng.randint( 16,T2-prefix-suffix ))
  idx.append( rng.randint( 16,T1-prefix-suffix ))
  idx.append( rng.randint( 16,T2-prefix-suffix ))
  idx.append( rng.randint( 16,T1-prefix-suffix ))
  idx.append( rng.randint( 16,T2-prefix-suffix ))
  return idx

def get_batch(data, actions, n):
    """Get a random batch of data from the specified bucket, prepare for step.
    Args
      data: a list of sequences of size n-by-d to fit the model to.
      actions: a list of the actions we are using
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """
    # Select entries at random
    for b in range(n):
      all_keys    = list(data.keys())
      chosen_keys = np.random.choice( len(all_keys), batch_size )
      # How many frames in total do we need?
      total_frames = seq_length_out
      decoder_outputs = np.zeros((batch_size, seq_length_out, input_size), dtype=float)

      for i in xrange( batch_size ):
        the_key = all_keys[ chosen_keys[i] ]
        # Get the number of frames
        n, _ = data[ the_key ].shape
        # Sample somewherein the middle
        idx = np.random.randint( 16, n-total_frames )
        # Select the data around the sampled points
        data_sel = data[ the_key ][idx:idx+total_frames ,:]
        decoder_outputs[i] = data_sel

      yield decoder_outputs

def get_batch_srnn(data, action, subject, subsequence):
  """
  Get a random batch of data from the specified bucket, prepare for step.

  Args
    data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
      v=nxd matrix with a sequence of poses
    action: the action to load data from
  Returns
    The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
    the constructed batches have the proper format to call step(...) later.
  """

  actions = ["directions", "discussion", "eating", "greeting", "phoning",
            "posing", "purchases", "sitting", "sittingdown", "smoking",
            "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

  if not action in actions:
    raise ValueError("Unrecognized action {0}".format(action))

  frames = {}
  # frames[ action ] = find_indices_srnn( data, action )

  batch_size = 1 # we always evaluate 8 seeds
  subject    = 5 # we always evaluate on subject 5
  source_seq_len = seq_length_in
  target_seq_len = seq_length_out

  # seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

  # encoder_inputs  = np.zeros( (batch_size, source_seq_len-1, input_size), dtype=float )
  # decoder_inputs  = np.zeros( (batch_size, target_seq_len, input_size), dtype=float )
  data_sel = data[ (subject, action, subsequence, 'even') ]
  decoder_outputs = np.zeros( (batch_size, len(data_sel), input_size), dtype=float )

  # Compute the number of frames needed
  # total_frames = source_seq_len + target_seq_len

  # Reproducing SRNN's sequence subsequence selection as done in
  # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
  for i in xrange( batch_size ):

    # _, _, idx = seeds[i]
    # idx = idx + 50


    # data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

    # encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
    # decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
    decoder_outputs[i, :, :] = data_sel

  # print(encoder_inputs.shape, decoder_inputs.shape, decoder_outputs.shape)
  # return encoder_inputs, decoder_inputs, decoder_outputs
  return decoder_outputs

def get_srnn_gts( actions, test_set, data_mean, data_std, dim_to_ignore, one_hot, subject, subsequence, to_euler=True ):
  """
  Get the ground truths for srnn's sequences, and convert to Euler angles.
  (the error is always computed in Euler angles).

  Args
    actions: a list of actions to get ground truths for.
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map

  Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
  """
  srnn_gts_euler = {}

  for action in actions:

    srnn_gt_euler = []
    # _, _, srnn_expmap = get_batch_srnn( test_set, action )
    srnn_expmap = get_batch_srnn( test_set, action, subject, subsequence )

    # expmap -> rotmat -> euler
    for i in np.arange( srnn_expmap.shape[0] ):
      denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          for k in np.arange(3,97,3):
            denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

      srnn_gt_euler.append( denormed );
    # Put back in the dictionary
    srnn_gts_euler[action] = np.array(srnn_gt_euler)

  return srnn_gts_euler

if __name__ == '__main__':
  action_list = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]

  relevant_coord = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

  print relevant_coord

  for action in action_list:
    actions = define_actions(action)

    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(actions, 
      seq_length_in, seq_length_out, '../../../../human-motion-prediction/data/h3.6m/dataset', False)

    for subject in [1,6,5,7,8,9,11]:
      for subsequence in [1,2]:
        srnn_gts_expmap = get_srnn_gts( actions, test_set, data_mean, data_std, dim_to_ignore, 
          not omit_one_hot, subject, subsequence, to_euler=False )

        data = np.zeros((len(srnn_gts_expmap[action][0]), len(relevant_coord)*3))
        for i in range(len(srnn_gts_expmap[action][0])):
          import matplotlib.pyplot as plt
          from mpl_toolkits.mplot3d import Axes3D
          pose = foward_kinematics.get_coords(srnn_gts_expmap[action][:,i])/1020
          pose = np.reshape(pose, (-1,3))
          data[i] = pose[relevant_coord].flatten()

          # END_I = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
          # END_J = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
          # fig = plt.figure()
          # ax = fig.add_subplot(111, projection='3d')
          # ax.set_xlim(-1, 1)
          # ax.set_ylim(-1, 1)
          # ax.set_zlim(-1, 1)
          # # xs = pose[:,0]
          # # ys = pose[:,1]
          # # zs = pose[:,2]
          # # for i in range(len(END_I)):
          # #   ax.plot(xs[[END_I[i], END_J[i]]], ys[[END_I[i], END_J[i]]], zs[[END_I[i], END_J[i]]], color='b')
          # # ax.scatter(xs[relevant_idx], ys[relevant_idx], zs[relevant_idx], color='r')
          # # ax.scatter(xs[-1:], ys[-1:], zs[-1:], color='b')
          # pose_ = np.reshape(data[i], (-1,3))
          # print pose_
          # xs = pose_[:,0]
          # ys = pose_[:,1]
          # zs = pose_[:,2]
          # ax.scatter(xs, ys, zs, color='r')
          # # ax.plot([0, xs[0]], [0, ys[0]], [0, zs[0]], color='r')
          # # ax.plot(xs[[10,13]], ys[[10,13]], zs[[10,13]], color='r')
          # plt.show()

          # foward_kinematics.plot(srnn_gts_expmap[action][i], srnn_gts_expmap[action][i])
        
        print (data.shape), np.max(data), np.min(data)
        assert np.max(data) < 1
        assert np.min(data) > -1

        np.save('../../h3.6/%s_%d_%d.npy'%(action, subject, subsequence), data)

else:
  import argparse
  import time
  import glob
  import os.path
  import numpy as np
  data_mean = None 
  data_std = None
  dim_to_ignore = None

  def get_model_load_name(model_name):
    return './models/%s_%d.hdf5'%(model_name, time.time())

  def get_log_name(model_name):
    return './models/%s_%d.log'%(model_name, time.time())

  def get_parse(model_name):
    global data_mean, data_std, dim_to_ignore

    ap = argparse.ArgumentParser()
    list_of_modes = ['train', 'sample']
    ap.add_argument('-m', '--mode', required=False, help='Choose between training mode or sampling mode.', default='train', choices=list_of_modes)
    ap.add_argument('-ep', '--epochs', required=False, help='Number of epochs', default='2', type=int)
    ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default='16', type=int)
    ap.add_argument('-lp', '--load_path', required=False, help='Model path', default=get_model_load_name(model_name))
    ap.add_argument('-t', '--timesteps', required=False, help='Timestep size', default='5', type=int)
    ap.add_argument('-p', '--periods', required=False, help='Number of iterations of the data', default='10', type=int)
    ap.add_argument('-ld', '--latent_dim', required=False, help='Embedding size', default='100', type=int)
    ap.add_argument('-o', '--option_dim', required=False, help='Number of options', default='2', type=int)
    ap.add_argument('-l', '--log_path', required=False, help='Log file for loss history', default=get_log_name(model_name))
    

    # ap.add_argument('-lr', '--learning_rate', required=False, help='Learning rate', default='5000', choices=list_of_modes)
    args = vars(ap.parse_args())
    
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(actions, 
      seq_length_in, seq_length_out, '../../../human-motion-prediction/data/h3.6m/dataset', False)

    step_per_epoch = 100
    validation_steps = 20
    train_data_gen = get_batch( train_set, not omit_one_hot, step_per_epoch)
    valid_data_gen = get_batch( test_set, not omit_one_hot, validation_steps)
    args['input_dim'], args['output_dim'] = input_size, input_size
    args['timesteps'] = seq_length_out
    return train_data_gen, valid_data_gen, step_per_epoch, validation_steps, args

  def plot_poses(batch_true, batch_predict, title, args):
    batch_true = data_utils.revert_output_format( batch_true, data_mean, data_std, dim_to_ignore, actions, not omit_one_hot )
    batch_predict = data_utils.revert_output_format( batch_predict, data_mean, data_std, dim_to_ignore, actions, not omit_one_hot )
    batch_true = np.reshape(batch_true, (batch_true.shape[0], -1))
    batch_predict = np.reshape(batch_predict, (batch_predict.shape[0], -1))
    foward_kinematics.plot_poses(batch_true, batch_predict, title, args)
