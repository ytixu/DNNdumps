import matplotlib
matplotlib.use('Agg')

import numpy as np

from utils import data_utils__
from utils import translate__

from utils import parser
from utils import image

MODEL_NAME = 'BASE_RNN'
HAS_LABELS = False

class seq2seq_ae__:

  def __init__(self, args):
    self.autoencoder = None
    self.encoder = None
    self.decoder = None

    self.epochs = args['epochs']
    self.batch_size = args['batch_size']
    # self.periods = args['periods']
    self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2
    self.lr = args['learning_rate']
    self.trained = False if args['mode'] == 'train' else True
    self.test_every = args['test_every']

    self.timesteps = args['timesteps'] if 'timesteps' in args else 5
    self.hierarchies = range(self.timesteps) if args['hierarchies'] == 'all' else args['hierarchies']
    self.conditioned_pred_steps = args['conditioned_pred_steps']
    self.latent_dim = args['latent_dim']

    if HAS_LABELS:
      self.label_dim = args['label_dim']
      self.data_dim = args['data_dim']+args['label_dim']
    else:
      self.data_dim = args['data_dim']

    self.labels = args['actions']
    self.has_labels = HAS_LABELS

    self.load_path = args['load_path']
    self.save_path = args['save_path']
    self.log_path = args['log_path']
    self.model_signature = args['model_signature']

    self.data_mean = args['data_mean']
    self.data_std = args['data_std']
    self.dim_to_ignore = args['dim_to_ignore']
    self.data_max = args['data_max']
    self.data_min = args['data_min']

  # def find_indices_srnn( self, data, action ):
  #   """
  #   Find the same action indices as in SRNN.
  #   See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
  #   """

  #   # Used a fixed dummy seed, following
  #   # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
  #   SEED = 1234567890
  #   rng = np.random.RandomState( SEED )

  #   subject = 5
  #   subaction1 = 1
  #   subaction2 = 2

  #   T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
  #   T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
  #   prefix, suffix = 50, 100

  #   idx = []
  #   idx.append( rng.randint( 16,T1-prefix-suffix ))
  #   idx.append( rng.randint( 16,T2-prefix-suffix ))
  #   idx.append( rng.randint( 16,T1-prefix-suffix ))
  #   idx.append( rng.randint( 16,T2-prefix-suffix ))
  #   idx.append( rng.randint( 16,T1-prefix-suffix ))
  #   idx.append( rng.randint( 16,T2-prefix-suffix ))
  #   idx.append( rng.randint( 16,T1-prefix-suffix ))
  #   idx.append( rng.randint( 16,T2-prefix-suffix ))
  #   print action,[idx[i] for i in [0,2,4,6]], [idx[i] for i in [1,3,5,7]], T1, T2
  #   return idx

  # def get_batch( self, data):
  #   """Get a random batch of data from the specified bucket, prepare for step.
  #   Args
  #     data: a list of sequences of size n-by-d to fit the model to.
  #   Returns
  #     batch_data: the constructed batch.
  #   """

  #   # Select entries at random
  #   all_keys    = list(data.keys())
  #   chosen_keys = np.random.choice( len(all_keys), self.batch_size )

  #   batch_data  = np.zeros((self.batch_size, self.timesteps, self.data_dim), dtype=float)

  #   for i in xrange( self.batch_size ):

  #     the_key = all_keys[ chosen_keys[i] ]

  #     # Get the number of frames
  #     n, _ = data[ the_key ].shape

  #     # Sample somewherein the middle
  #     idx = np.random.randint( 16, n-self.timesteps )

  #     # Select the data around the sampled points
  #     batch_data[i] = data[ the_key ][idx:idx+self.timesteps ,:]

  #   return batch_data


  # def get_batch_srnn(self, data, action ):
  #   """
  #   Get a random batch of data from the specified bucket, prepare for step.
  #   Args
  #     data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
  #       v=nxd matrix with a sequence of poses
  #     action: the action to load data from
  #   Returns
  #     batch_data : the constructed batches have the proper format to call step(...) later.
  #   """
  #   print action, type(action)
  #   actions = ["directions", "discussion", "eating", "greeting", "phoning",
  #             "posing", "purchases", "sitting", "sittingdown", "smoking",
  #             "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

  #   if not action in actions:
  #     raise ValueError("Unrecognized action {0}".format(action))

  #   frames = {}
  #   frames[ action ] = self.find_indices_srnn( data, action )

  #   batch_size = 8 # we always evaluate 8 seeds
  #   subject    = 5 # we always evaluate on subject 5

  #   seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

  #   batch_data  = np.zeros( (batch_size, self.timesteps, self.data_dim), dtype=float )

  #   # Reproducing SRNN's sequence subsequence selection as done in
  #   # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
  #   for i in xrange( batch_size ):

  #     _, subsequence, idx = seeds[i]
  #     idx = idx + 50

  #     data_sel = data[ (subject, action, subsequence, 'even') ]

  #     data_sel = data_sel[(idx-self.conditioned_pred_steps):(idx+self.timesteps-self.conditioned_pred_steps) ,:]

  #     batch_data[i]  = data_sel

  #   return batch_data

  def training_images_plotter(self, pred_data, gt_data):
    xyz_pred = translate__.batch_expmap2xyz(pred_data, self)
    xyz_gt = translate__.batch_expmap2xyz(gt_data, self)
    image.plot_poses(xyz_pred[range(0,xyz_pred.shape[0],xyz_pred.shape[0]/5)][:,range(0,self.timesteps,self.timesteps/5)],
                      np.array([xyz_gt[:,range(0,self.timesteps,self.timesteps/5)]]))

if __name__ == '__main__':
  train_set, test_set, config = parser.get_parse(MODEL_NAME, HAS_LABELS, create_params=True)
  ae = seq2seq_ae__(config)
  #test_gt, test_pred_gt = test_set


  '''
  test conversions
  '''
  #for x in train_set:
  #  xyz = translate__.batch_expmap2xyz(x[:5,:5], ae)
  #  image.plot_poses(xyz)
  #  break

  batch_data = ae.get_batch( train_set)
  print 'train batch', batch_data.shape
  xyz = translate__.batch_expmap2xyz(batch_data[:5, :5], ae)
  image.plot_poses(xyz)

  #for action in config['actions']:
  #   batch_data = ae.get_batch_srnn( test_set, action)
  #   print 'test batch', batch_data.shape

  # #   xyz = translate__.batch_expmap2xyz(batch_data, ae)
  # #   image.plot_poses(xyz)

  #   print translate__.euler_diff(batch_data, batch_data, ae)

def other_():
  '''
  test prediction from Martinez et al.
  load generated predictions from sample.h5
  + sanity check in the conversions from translate__.py
  '''

  import h5py
  # numpy implementation
  n = ae.timesteps-ae.conditioned_pred_steps
  expmap_gt = np.zeros((8, n, 99))
  expmap_pred = np.zeros((8, n, 99))
  with h5py.File( '../baselines/samples.h5', 'r' ) as h5f:
    for i, action in enumerate(config['actions']):
      for j in range(8):
        expmap_gt[j] = h5f['expmap/preds_gt/%s_%d'%(action, j)][:n]
        expmap_pred[j] = h5f['expmap/preds/%s_%d'%(action, j)][:n]

      # batch_data = ae.get_batch_srnn( test_set, action)
      # print batch_data.shape
      print [i*8+k for k in [0,4,1,5,2,6,3,7]]
      loaded_batch = test_pred_gt[[i*8+k for k in [0,4,1,5,2,6,3,7]],:n]
      xyz = translate__.batch_expmap2xyz(loaded_batch, ae)
      image.plot_poses(xyz[:,:5])
      #xyz_p = translate__.batch_expmap2xyz(expmap_gt[:,:5], ae, normalized=False)
      #image.plot_poses(xyz[:,:5])

      print action, n
      print translate__.euler_diff(expmap_gt, expmap_pred, ae, normalized=[False, False])[0]
      #print translate__.euler_diff(expmap_gt, batch_data[:,-n:], ae, normalized=[False, True])
      print translate__.euler_diff(expmap_gt, loaded_batch, ae, normalized=[False, True])[0]
      #break

