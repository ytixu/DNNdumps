"""Functions that help with data processing for human3.6m"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import copy
import csv
import math

def rotmat2euler( R ):
  """
  Converts a rotation matrix to Euler angles
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1
  Args
    R: a 3x3 rotation matrix
  Returns
    eul: a 3x1 Euler angle representation of R
  """
  if R[0,2] == 1 or R[0,2] == -1:
    # special case
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;

  else:
    E2 = -np.arcsin( R[0,2] )
    E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul


def quat2expmap(q):
  """
  Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
  Args
    q: 1x4 quaternion
  Returns
    r: 1x3 exponential map
  Raises
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  if (np.abs(np.linalg.norm(q)-1)>1e-3):
    raise(ValueError, "quat2expmap: input quaternion is not norm 1")

  sinhalftheta = np.linalg.norm(q[1:])
  coshalftheta = q[0]

  r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
  theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
  theta = np.mod( theta + 2*np.pi, 2*np.pi )

  if theta > np.pi:
    theta =  2 * np.pi - theta
    r0    = -r0

  r = r0 * theta
  return r

def rotmat2quat(R):
  """
  Converts a rotation matrix to a quaternion
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4
  Args
    R: 3x3 rotation matrix
  Returns
    q: 1x4 quaternion
  """
  rotdiff = R - R.T;

  r = np.zeros(3)
  r[0] = -rotdiff[1,2]
  r[1] =  rotdiff[0,2]
  r[2] = -rotdiff[0,1]
  sintheta = np.linalg.norm(r) / 2;
  r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );

  costheta = (np.trace(R)-1) / 2;

  theta = np.arctan2( sintheta, costheta );

  q      = np.zeros(4)
  q[0]   = np.cos(theta/2)
  q[1:] = r0*np.sin(theta/2)
  return q

def rotmat2expmap(R):
  return quat2expmap( rotmat2quat(R) );

def expmap2rotmat(r):
  """
  Converts an exponential map angle to a rotation matrix
  Matlab port to python for evaluation purposes
  I believe this is also called Rodrigues' formula
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
  Args
    r: 1x3 exponential map
  Returns
    R: 3x3 rotation matrix
  """
  theta = np.linalg.norm( r )
  r0  = np.divide( r, theta + np.finfo(np.float32).eps )
  r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
  r0x = r0x - r0x.T
  R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);
  return R


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot, data_max=0, data_min=0):
  """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
  Args
    normalizedData: nxd matrix with normalized data
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    origData: data originally used to
  """
  # redo normalization between -1 and 1
  # if data_max > 0:
  #  if one_hot:
  #    normalizedData[:,:-len(actions)] = (normalizedData[:,:-len(actions)] + 1)/2*(data_max-data_min)+data_min
  #  else:
  #    normalizedData = (normalizedData+1)/2*(data_max-data_min) + data_min

  T = normalizedData.shape[0]
  D = data_mean.shape[0]

  origData = np.zeros((T, D), dtype=np.float32)
  dimensions_to_use = []
  for i in range(D):
    if i in dimensions_to_ignore:
      continue
    dimensions_to_use.append(i)
  dimensions_to_use = np.array(dimensions_to_use)

  if one_hot:
    origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
  else:
    origData[:, dimensions_to_use] = normalizedData

  # potentially ineficient, but only done once per experiment
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(origData, stdMat) + meanMat
  return origData


# def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
#   """
#   Converts the output of the neural network to a format that is more easy to
#   manipulate for, e.g. conversion to other format or visualization
#   Args
#     poses: The output from the TF model. A list with (seq_length) entries,
#     each with a (batch_size, dim) output
#   Returns
#     poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
#     batch is an n-by-d sequence of poses.
#   """
#   seq_len = len(poses)
#   if seq_len == 0:
#     return []

#   batch_size, dim = poses[0].shape

#   poses_out = np.concatenate(poses)
#   poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
#   poses_out = np.transpose(poses_out, [1, 0, 2])

#   poses_out_list = []
#   for i in xrange(poses_out.shape[0]):
#     poses_out_list.append(
#       unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

#   return poses_out_list


def readCSVasFloat(filename):
  """
  Borrowed from SRNN code. Reads a csv and returns a float matrix.
  https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34
  Args
    filename: string. Path to the csv file
  Returns
    returnArray: the read data in a float32 matrix
  """
  returnArray = []
  lines = open(filename).readlines()
  for line in lines:
    line = line.strip().split(',')
    if len(line) > 0:
      returnArray.append(np.array([np.float32(x) for x in line]))

  returnArray = np.array(returnArray)
  return returnArray

def readCSVasFloat_randLines(filename, timesteps, rand_n, one_hot, action_n):
  with open(filename, 'r') as csvfile:
    lines = np.array(list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)))
    # skip every second frame
    line_n = int(math.ceil(lines.shape[0]/2.0))-16-timesteps
    rand_n = min(line_n,rand_n)
    # Sample somewherein the middle (from seq2seq_model.get_batch)
    line_idx = np.random.choice(line_n, rand_n, replace=False)+16
    data_dim = lines[0].shape[-1]
    if one_hot:
      returnArray = np.zeros((rand_n, timesteps, data_dim+action_n))
    else:
      returnArray = np.zeros((rand_n, timesteps, data_dim))

    for i, idx in enumerate(line_idx):
      returnArray[i,:,:data_dim] = lines[range(2*idx,2*(idx+timesteps),2)]

  return returnArray

# (from seq2seq_model.find_indices_srnn)
def find_indices_srnn( action, subj ):
    """
    Find the same action indices as in SRNN.
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """
    '''
    Hard copy of the indices as produced in seq2seq_model.find_indices_srnn
    '''
    return {'walking' : [[1087, 1145, 660, 201],[955, 332, 304, 54]],
    'eating' : [[1426, 1087, 1329, 1145],[374, 156, 955, 332]],
    'smoking' : [[1426, 1087, 1329, 1145],[1398, 1180, 955, 332]],
    'discussion' : [[1426, 1398, 1180, 332],[2063, 1087, 1145, 1438]],
    'directions' : [[1426, 1087, 1145, 1438],[374, 156, 332, 665]],
    'greeting' : [[402, 63, 305, 121],[1398, 1180, 955, 332]],
    'phoning' : [[1426, 1087, 1329, 332],[374, 156, 121, 414]],
    'posing' : [[402, 63, 835, 955],[374, 156, 305, 121]],
    'purchases' : [[1087, 955, 332, 304],[1180, 1145, 660, 201]],
    'sitting' : [[1426, 1087, 1329, 1145],[1398, 1180, 955, 332]],
    'sittingdown' : [[1426, 1087, 1145, 1438],[1398, 1180, 332, 1689]],
    'takingphoto' : [[1426, 1180, 1145, 1438],[1087, 955, 332, 660]],
    'waiting' : [[1426, 1398, 1180, 332],[2063, 1087, 1145, 1438]],
    'walkingdog' : [[402, 63, 305, 332],[374, 156, 121, 414]],
    'walkingtogether' : [[1087, 1329, 1145, 660],[1180, 955, 332, 304]]}[action][subj-1]

def readCSVasFloat_for_validation(filename, action, subact, one_hot):

  with open(filename, 'r') as csvfile:
    lines = np.array(list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)))
    data_dim = lines.shape[-1]

    # (from seq2seq_model.get_batch_srnn)
    frames = find_indices_srnn( action, subact )

    if one_hot:
      returnArray = np.zeros((len(frames), 150, data_dim+action_n))
    else:
      returnArray = np.zeros((len(frames), 150, data_dim))

    # 150 frames (as in seq2seq_model.get_batch_srnn)
    for i, idx in enumerate(frames):
      # skip every second frame
      returnArray[i,:,:data_dim] = lines[range(2*idx,2*(idx+150),2)][:]

  return returnArray


def load_data(path_to_dataset, subjects, actions, one_hot, generator=False):
  """
  Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270
  Args
    path_to_dataset: string. directory where the data resides
    subjects: list of numbers. The subjects to load
    actions: list of string. The actions to load
    one_hot: Whether to add a one-hot encoding to the data
  Returns
    trainData: dictionary with k:v
      k=(subject, action, subaction, 'even'), v=(nxd) un-normalized data
    completeData: nxd matrix with all the data. Used to normlization stats
  """
  nactions = len( actions )

  trainData = {}
  completeData = []
  for subj in subjects:
    for action_idx in np.arange(len(actions)):

      action = actions[ action_idx ]

      for subact in [1, 2]:  # subactions

        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

        filename = '{0}/S{1}/{2}_{3}.txt'.format( path_to_dataset, subj, action, subact)
        action_sequence = readCSVasFloat(filename)

        n, d = action_sequence.shape
        even_list = range(0, n, 2)

        if one_hot:
          # Add a one-hot encoding at the end of the representation
          the_sequence = np.zeros( (len(even_list), d + nactions), dtype=float )
          the_sequence[ :, 0:d ] = action_sequence[even_list, :]
          the_sequence[ :, d+action_idx ] = 1

          #if generator:
          #  yield (subj, action, subact), the_sequence
          #  continue

          trainData[(subj, action, subact, 'even')] = the_sequence
        else:

          #if generator:
          #  yield (subj, action, subact), action_sequence[even_list, :]
          #  continue

          trainData[(subj, action, subact, 'even')] = action_sequence[even_list, :]


        if len(completeData) == 0:
          completeData = copy.deepcopy(action_sequence)
        else:
          completeData = np.append(completeData, action_sequence, axis=0)
	break
      break
    break

  return trainData, completeData

def load_data_(path_to_dataset, subjects, actions, action_n, one_hot, func):
  '''
  Helper function for loading data
  '''
  data_sequences = []
  for subj in subjects:
    for action_idx in np.arange(action_n):
      action = actions[ action_idx ]
      for subact in [1, 2]:  # subactions
        # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
        filename = '{0}/S{1}/{2}_{3}.txt'.format( path_to_dataset, subj, action, subact)

        action_sequences = func(filename, action, subact)

        if one_hot:
          # Add a one-hot encoding at the end of the representation
          action_sequences[ :, :, action_idx-action_n ] = 1

        if len(data_sequences) == 0:
          data_sequences = action_sequences
        else:
          data_sequences = np.concatenate([data_sequences, action_sequences], axis=0)
  return data_sequences

def load_rand_data(path_to_dataset, subjects, actions, one_hot, timesteps, rand_n, iter_n):
  action_n = len(actions)
  rand_n = int(rand_n/action_n/2/len(subjects))
  func = lambda filename, action, subact : readCSVasFloat_randLines(filename, timesteps, rand_n, one_hot, action_n)

  while iter_n > 0:
    data_sequences = load_data_(path_to_dataset, subjects, actions, action_n, one_hot, func)
    iter_n -= 1
    yield data_sequences

def get_test_data(path_to_dataset, subjects, actions, one_hot):
  '''
  beware that the ordering of the motion sequence for each action is different
  '''
  action_n = len(actions)
  func = lambda filename, action, subact : readCSVasFloat_for_validation(filename, action, subact, one_hot)
  data_sequences = load_data_(path_to_dataset, subjects, actions, action_n, one_hot, func)
  print ('test data shape', data_sequences.shape)
  return data_sequences[:,:50], data_sequences[:,50:]

def normalize_data( data, data_mean, data_std, dim_to_use, actions, one_hot, data_max=0, data_min=0 ):
  """
  Normalize input data by removing unused dimensions, subtracting the mean and
  dividing by the standard deviation
  Args
    data: nx99 matrix with data to normalize
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dim_to_use: vector with dimensions used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    data_out: the passed data matrix, but normalized
  """
  nactions = len(actions)

  if type(data) == type({}):
    data_out = {}
    if not one_hot:
      # No one-hot encoding... no need to do anything special
      for key in data.keys():
        data_out[ key ] = np.divide( (data[key] - data_mean), data_std )
        data_out[ key ] = data_out[ key ][ :, dim_to_use ]

    else:
      # TODO hard-coding 99 dimensions for un-normalized human poses
      for key in data.keys():
        data_out[ key ] = np.divide( (data[key][:, 0:99] - data_mean), data_std )
        data_out[ key ] = data_out[ key ][ :, dim_to_use ]
        data_out[ key ] = np.hstack( (data_out[key], data[key][:,-nactions:]) )

    return data_out

  else:
    if one_hot:
      dim_to_use = dim_to_use + range(99,99+nactions)
    data[:,:,:99] = np.divide( (data[:,:,:99] - data_mean), data_std )
    # normalize between -1 and 1
    # data[:,:,:99] = 2*(data[:,:,:99]-data_min)/(data_max-data_min) - 1
    return data[ :,:,dim_to_use ]

def normalization_stats(completeData):
  """"
  Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33
  Args
    completeData: nx99 matrix with data to normalize
  Returns
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    dimensions_to_use: vector with dimensions used by the model
  """
  data_mean = np.mean(completeData, axis=0)
  data_std  =  np.std(completeData, axis=0)

  dimensions_to_ignore = []
  dimensions_to_use    = []

  dimensions_to_ignore.extend( list(np.where(data_std < 1e-4)[0]) )
  dimensions_to_use.extend( list(np.where(data_std >= 1e-4)[0]) )

  data_std[dimensions_to_ignore] = 1.0

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use
