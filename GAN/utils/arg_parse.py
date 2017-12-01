import argparse
import data_reader

def get_parse():
	ap = argparse.ArgumentParser()
	#list_of_methods = ['wordnet', 'word2vec', 'onehot', 'glove']
	list_of_modes = ['train', 'sample']
	#ap.add_argument('-m', '--method', required=False, help='Method to use for WSD. Default = wordnet.', default='wordnet', choices = list_of_methods)
	ap.add_argument('-d', '--data', required=True, help='Training data file.')
	ap.add_argument('-lf', '--load_file', required=False, help='Filename selected for loading trained model.')
	ap.add_argument('-m', '--mode', required=False, help='Choose between training mode or sampling mode.', default='train', choices=list_of_modes)
	ap.add_argument('-ep', '--epoch', required=False, help='Number of epoch', default='5000')
	ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default='16')
	# ap.add_argument('-lr', '--learning_rate', required=False, help='Learning rate', default='5000', choices=list_of_modes)

	args = vars(ap.parse_args())
	data = data_reader.read(args['data'])

	return data, args
