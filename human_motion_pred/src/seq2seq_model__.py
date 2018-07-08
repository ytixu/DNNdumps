import matplotlib
matplotlib.use('Agg')

import numpy as np

from utils import data_utils__
from utils import translate__

from utils import parser
from utils import image

MODEL_NAME = 'BASE_RNN'

class seq2seq_ae__:

  def __init__(self, args):
    self.autoencoder = None
    self.encoder = None
    self.decoder = None

    self.epochs = args['epochs']
    self.batch_size = args['batch_size']
    self.periods = args['periods']
    self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2
    self.lr = args['learning_rate']

    self.timesteps = args['timesteps'] if 'timesteps' in args else 5
    self.hierarchies = args['hierarchies']
    self.conditioned_pred_steps = self.timesteps - 10
    self.latent_dim = args['latent_dim']
    self.label_dim = args['label_dim']
    self.data_dim = args['data_dim']+args['label_dim']
    self.labels = args['labels']

    self.load_path = args['load_path']
    self.save_path = args['save_path']
    self.log_path = args['log_path']
    self.model_signature = args['model_signature']


  def make_model(self):
    pass


  def get_batch( self, data):
    """Get a random batch of data from the specified bucket, prepare for step.
    Args
      data: a list of sequences of size n-by-d to fit the model to.
    Returns
      batch_data: the constructed batch.
    """

    # Select entries at random
    all_keys    = list(data.keys())
    chosen_keys = np.random.choice( len(all_keys), self.batch_size )

    batch_data  = np.zeros((self.batch_size, self.timesteps, self.data_dim), dtype=float)

    for i in xrange( self.batch_size ):

      the_key = all_keys[ chosen_keys[i] ]

      # Get the number of frames
      n, _ = data[ the_key ].shape

      # Sample somewherein the middle
      idx = np.random.randint( 16, n-self.timesteps )

      # Select the data around the sampled points
      batch_data = data[ the_key ][idx:idx+self.timesteps ,:]

    return batch_data


  def get_batch_srnn(self, data, action ):
    """
    Get a random batch of data from the specified bucket, prepare for step.
    Args
      data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
        v=nxd matrix with a sequence of poses
      action: the action to load data from
    Returns
      batch_data : the constructed batches have the proper format to call step(...) later.
    """

    actions = ["directions", "discussion", "eating", "greeting", "phoning",
              "posing", "purchases", "sitting", "sittingdown", "smoking",
              "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

    if not action in actions:
      raise ValueError("Unrecognized action {0}".format(action))

    frames = {}
    frames[ action ] = self.find_indices_srnn( data, action )

    batch_size = 8 # we always evaluate 8 seeds
    subject    = 5 # we always evaluate on subject 5

    seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

    batch_data  = np.zeros( (batch_size, self.timesteps, self.data_dim), dtype=float )

    # Reproducing SRNN's sequence subsequence selection as done in
    # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    for i in xrange( batch_size ):

      _, subsequence, idx = seeds[i]
      idx = idx + 50

      data_sel = data[ (subject, action, subsequence, 'even') ]

      data_sel = data_sel[(idx-self.conditioned_pred_steps):(idx+self.timesteps-self.conditioned_pred_steps) ,:]

      batch_data[i]  = data_sel

    return batch_data


if __name__ == '__main__':
  # test conversions
  labels=True

  train_set, test_set, config = parser.get_parse(NAME, labels)
  print train_set.values()[0].shape

  ae = seq2seq_ae__(config)
  batch_data = ae.get_batch( train_set)
  print batch_data.shape

  batch_data = ae.get_batch_srnn( test_set, config['actions'])
  print batch_data.shape

  xyz = translate__.batch_expmap2euler()
  image.plot_poses(xyz)