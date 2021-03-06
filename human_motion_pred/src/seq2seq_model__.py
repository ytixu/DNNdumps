import matplotlib
matplotlib.use('Agg')

import csv
import numpy as np
from sklearn import cross_validation

from utils import data_utils__
from utils import translate__

from utils import parser
from utils import image
from utils import metrics

MODEL_NAME = 'BASE_RNN'
HAS_LABELS = False

class seq2seq_ae__:

  def __init__(self, args, has_labels):
    self.autoencoder = None
    self.encoder = None
    self.decoder = None

    self.epochs = args['epochs']
    self.batch_size = args['batch_size']
    # self.periods = args['periods']
    self.decay = args['decay']
    self.decay_numb = int(args['periods']/args['decay_times'])
    self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2
    self.lr = args['learning_rate']
    self.trained = args['mode'] == 'sample'
    self.test_every = args['test_every']

    self.timesteps = args['timesteps'] if 'timesteps' in args else 5
    self.hierarchies = range(self.timesteps) if args['hierarchies'] == 'all' else map(int, args['hierarchies'])
    print self.hierarchies
    self.conditioned_pred_steps = args['conditioned_pred_steps']
    self.latent_dim = args['latent_dim']

    if has_labels:
      self.label_dim = args['label_dim']
      self.motion_dim = args['data_dim']
      self.data_dim = args['data_dim']+args['label_dim']
    else:
      self.data_dim = args['data_dim']

    self.labels = args['actions']
    self.has_labels = has_labels

    self.load_path = args['load_path']
    self.save_path = args['save_path']
    self.log_path = args['log_path']
    self.model_signature = args['model_signature']

    self.data_mean = args['data_mean']
    self.data_std = args['data_std']
    self.dim_to_ignore = args['dim_to_ignore']
    self.data_max = args['data_max']
    self.data_min = args['data_min']

    self.loss_count = 1000
    self.iter_count = 0

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

  def make_model(self):
    pass

  def recompile_opt(self):
    pass

  def load(self, path=''):
    if path == '':
      path = self.load_path
    if path != '':
      self.autoencoder.load_weights(path)
      print 'LOADED', path
      return True
    return False

  def __post_train_step(self, new_loss, x_test, rec=[], n=25):
    # print new_loss
    if new_loss < self.loss_count:
      self.autoencoder.save_weights(self.save_path, overwrite=True)
      self.loss_count = new_loss
      print 'Saved model -', new_loss, self.save_path

    if self.iter_count % self.test_every == 0:
      #y_test_decoded = self.autoencoder.predict(x_test[:1])
      #y_test_decoded = np.reshape(y_test_decoded, (len(self.hierarchies), self.timesteps, -1))
      #self.__training_images_plotter(y_test_decoded, x_test[:1])

      idx = np.random.choice(x_test.shape[0], n, replace=False)
      y_test_encoded = self.encoder.predict(x_test[idx])
      log_err = [new_loss, 0, 0]
      for h, cut in enumerate([self.conditioned_pred_steps, self.timesteps]):
        y_test_decoded = self.decoder.predict(y_test_encoded[:,cut-1])
        euler_err = translate__.euler_diff(x_test[idx,:cut], y_test_decoded[:,:cut], self)[0]
        print euler_err
        log_err[h+1] = np.mean(euler_err)

      self.__log_training_error(log_err+rec)
    else:
      self.__log_training_error([new_loss]+rec)

    self.iter_count += 1

    if self.iter_count % self.decay_numb == 0:
      self.load(self.save_path)
      self.lr = self.lr*self.decay
      self.recompile_opt()

    print 'Iteration -', self.iter_count

  def __training_images_plotter(self, pred_data, gt_data):
    xyz_pred = translate__.batch_expmap2xyz(pred_data, self)
    xyz_gt = translate__.batch_expmap2xyz(gt_data, self)
    image.plot_poses(xyz_pred[range(0,xyz_pred.shape[0],xyz_pred.shape[0]/5)][:,range(0,self.timesteps,self.timesteps/5)],
                      np.array([xyz_gt[:,range(0,self.timesteps,self.timesteps/5)]]))

  def __log_training_error(self, log):
    with open(self.log_path, 'a+') as f:
      spamwriter = csv.writer(f)
      spamwriter.writerow(log + [self.lr])

  def __alter_y(self, y):
    if len(self.hierarchies) == 1:
      return y
    y = np.repeat(y, len(self.hierarchies), axis=0)
    y = np.reshape(y, (-1, len(self.hierarchies), self.timesteps, y.shape[-1]))
    for i, h in enumerate(self.hierarchies):
      for j in range(h+1, self.timesteps):
        y[:,i,j] = y[:,i,h]
    return np.reshape(y, (-1, self.timesteps*len(self.hierarchies), y.shape[-1]))

  def __alter_label(self, x):
    idx = np.random.choice(x.shape[0], x.shape[0]/2, replace=False)
    x[idx,:,-self.label_dim:] = 0
    return x

  def run(self, data_iterator, test_data_func, has_labels, args=[]):
    self.make_model()
    self.load()
    if not self.trained:
      # from keras.utils import plot_model
      # plot_model(self.autoencoder, to_file='model.png')
      for x in data_iterator:
        if has_labels:
          x = self.__alter_label(x)

        x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, x, test_size=self.cv_splits)
        y_train = self.__alter_y(y_train)
        y_test = self.__alter_y(y_test)
        print x_train.shape, x_test.shape, y_train.shape, y_test.shape
        # from utils import image
        # image.plot_data(x_train[0])
        #xyz = translate__.batch_expmap2xyz(y_train[:5,:5], self)
        #image.plot_poses(xyz)

        history = self.autoencoder.fit(x_train, y_train,
              shuffle=True,
              epochs=self.epochs,
              batch_size=self.batch_size,
              validation_data=(x_test, y_test))

        self.__post_train_step(history.history['loss'][0], x_test, args)

    else:
      # testing
      test_x, test_y = test_data_func()
      embedding = metrics.load_embedding(self, data_iterator, [self.conditioned_pred_steps-1, self.timesteps-1])
      metrics.get_mesures(self, embedding, test_x, test_y)


def convert__(ae, train_set, test_set):
  max_ = 0

  train_set, train_set_ = tee(train_set)
  test_set, test_set_ = tee(test_set)

  for data_set in [train_set, test_set]:
    for k, x in data_set:
      print x.shape
      xyz = translate__.batch_expmap2xyz(np.array([x]), ae)[0]
      print xyz[:, :3]
      #image.plot_poses([xyz[1157:1162], xyz[5:10], xyz[10:15]])
      new_max = np.max(np.abs(xyz))
      if new_max > max_:
        max_ = new_max

  return max_
  for data_name, data_set in ({'train':train_set_, 'valid':test_set_}).iteritems():
    for k, x in data_set:
      print 'save', k, x.shape
      subject, action, subact = k
      xyz = np.array(translate__.batch_expmap2euler(np.array([x]), ae)[0])
      #np.save('../../data/h3.6/full/%s/%s_%d_%d.npy'%(data_name, action, subject, subact), xyz)
      #np.save('../../data/h3.6/full/%s_expmap/%s_%d_%d.npy'%(data_name, action, subject, subact), x)
      print xyz.shape
      np.save('../../data/h3.6/full/%s_euler/%s_%d_%d.npy'%(data_name, action, subject, subact), xyz)

      #print np.min(xyz), np.max(xyz)


if __name__ == '__main__':
  from itertools import tee
  train_set, test_set, config = parser.get_parse(MODEL_NAME, HAS_LABELS)
  ae = seq2seq_ae__(config, HAS_LABELS)
  #test_gt, test_pred_gt = test_set

  max_ = convert__(ae, train_set, test_set)
  print max_
  #max_ = 1.0

  expmap_cond_all, expmap_pred_gt = test_set()

  '''
  test conversions
  '''
  #for x in train_set:
  #  xyz = translate__.batch_expmap2xyz(x[:5,:5], ae)
  #  image.plot_poses(xyz)
  #  break

  # batch_data = ae.get_batch( train_set)
  # print 'train batch', batch_data.shape
  # xyz = translate__.batch_expmap2xyz(batch_data[:5, :5], ae)
  # image.plot_poses(xyz)

  #for action in config['actions']:
  #   batch_data = ae.get_batch_srnn( test_set, action)
  #   print 'test batch', batch_data.shape

  # #   xyz = translate__.batch_expmap2xyz(batch_data, ae)
  # #   image.plot_poses(xyz)

  #   print translate__.euler_diff(batch_data, batch_data, ae)

  '''
  test prediction from Martinez et al.
  load generated predictions from sample.h5
  + sanity check in the conversions from translate__.py
  '''

  import h5py
  # numpy implementation
  n = 100 # ae.timesteps - ae.conditioned_pred_steps
  #expmap_cond = np.zeros((8, n, 99))
  expmap_gt = np.zeros((8, n, 99))
  expmap_pred = np.zeros((8, n, 99))
  with h5py.File( '../baselines/samples.h5', 'r' ) as h5f:
    for i, action in enumerate(config['actions']):
      for j in range(8):
        #expmap_cond[j] = h5f['expmap/gt/%s_%d'%(action, j)] # [:n/2]
        expmap_gt[j] = h5f['expmap/preds_gt/%s_%d'%(action, j)] # [:n]
        expmap_pred[j] = h5f['expmap/preds/%s_%d'%(action, j)] # [:n]

      #batch_data = ae.get_batch_srnn( test_set, action)
      #print batch_data.shape
      #print [i*8+k for k in [0,4,1,5,2,6,3,7]]
      #loaded_batch = test_pred_gt[[i*8+k for k in [0,4,1,5,2,6,3,7]],:n]
      #xyz = translate__.batch_expmap2xyz(loaded_batch, ae)
      #image.plot_poses(xyz[:,:5])
      #xyz_p = translate__.batch_expmap2xyz(expmap_gt[:,:5], ae, normalized=False)
      #image.plot_poses(xyz[:,:5])

      #loaded_batch = expmap_pred_gt[[i*8+k for k in [0,4,1,5,2,6,3,7]]]
      #error = translate__.euler_diff(loaded_batch[:,:50], expmap_gt[:,:50], ae, normalized=[True, False])[0]
      #print error
      expmap_cond = expmap_cond_all[[i*8+k for k in [0,4,1,5,2,6,3,7]]]

      print action, n
      error = translate__.euler_diff(expmap_gt, expmap_pred, ae, normalized=[False, False])[0]
      continue
      #print error

      euler_cond = np.array(translate__.batch_expmap2xyz(expmap_cond, ae, normalized=True))
      euler_gt = np.array(translate__.batch_expmap2xyz(expmap_gt, ae, normalized=False))
      euler_pred = np.array(translate__.batch_expmap2xyz(expmap_pred, ae, normalized=False))
      print np.mean(np.abs(euler_cond[:,-1] - euler_gt[:,0])), np.mean(np.abs(euler_cond[:,-1] - euler_cond[:,-2]))

      # euler_gt[0][:,0:6] = 0
      # idx_to_use = np.where( np.std(euler_gt[0], 0 ) > 1e-4 )[0]
      #idx_to_use = [6,7,8,9,12,13,14,15,21,22,23,24,27,28,29,30,36,37,38,39,40,41,42,43,44,45,46,47,51,52,53,54,55,56,57,60,61,62,75,76,77,78,79,80,81,84,85,86]

      #euc_error = np.power(euler_gt[:,:,idx_to_use] - euler_pred[:,:,idx_to_use], 2)
      #euc_error = np.sum(euc_error, -1)
      #euc_error = np.mean(np.sqrt(euc_error), 0)
      #print euc_error - error

      np.save('../baselines/xyz/%s_gt.npy'%(action), euler_gt/max_)
      np.save('../baselines/xyz/%s_cond.npy'%(action), euler_cond/max_)
      np.save('../baselines/xyz/%s_pred.npy'%(action), euler_pred/max_)

      #print translate__.euler_diff(expmap_gt, batch_data[:,-n:], ae, normalized=[False, True])
      #print translate__.euler_diff(expmap_gt, loaded_batch, ae, normalized=[False, True])[0]
      #break


