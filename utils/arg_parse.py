import argparse
import data_reader

def get_parse():
	ap = argparse.ArgumentParser()
	#list_of_methods = ['wordnet', 'word2vec', 'onehot', 'glove']
	list_of_modes = ['train', 'sample']
	#ap.add_argument('-m', '--method', required=False, help='Method to use for WSD. Default = wordnet.', default='wordnet', choices = list_of_methods)
	ap.add_argument('-d', '--data', required=True, help='Training data file.')
	ap.add_argument('-r', '--reference', required=True, help='References data file.')
	ap.add_argument('-m', '--mode', required=False, help='Choose between training mode or sampling mode.', default='train', choices=list_of_modes)
	ap.add_argument('-ep', '--epochs', required=False, help='Number of epochs', default='55', type=int)
	ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default='100', type=int)
	ap.add_argument('-ih', '--image_height', required=False, help='Image height', default='480', type=int)
	ap.add_argument('-iw', '--image_width', required=False, help='Image width', default='720', type=int)
	ap.add_argument('-lp', '--load_path', required=False, help='Model path', default='DNNdumps/models/autoencoder.hdf5')
	# ap.add_argument('-ls', '--latent_size', required=False, help='Latent vector size * n = intput size', default='2', type=int)
	# ap.add_argument('-lr', '--learning_rate', required=False, help='Learning rate', default='5000', choices=list_of_modes)

	args = vars(ap.parse_args())
	x, y = data_reader.read(args['data'], args['reference'], args['image_height'], args['image_width'])

	return x, y, args
