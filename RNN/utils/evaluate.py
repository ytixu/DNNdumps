import numpy as np
from tqdm import tqdm
import metrics

def error(y_true, y_predict):
	return np.mean(np.mean(np.square(y_true - y_predict), axis=1), axis=1)

def random_baseline(data_iterator):

	x_ref = []
	x_pick = []
	h = 0
	for x, y in data_iterator:
		x_ref_ = x[np.random.choice(len(x), 10)]
		x_pick_ = x[np.random.choice(len(x), 10)]
		if len(x_ref) == 0:
			x_ref = x_ref_
			x_pick = x_pick_
			h = x_ref.shape[1]
		else:
			x_ref = np.concatenate((x_ref, x_ref_), axis=0)
			x_pick = np.concatenate((x_pick, x_pick_), axis=0)

	print h
	mean = np.zeros(h)
	var = np.zeros(h)
	for i in range(h):
		x_ref[:,:i+1] = 0
		x_pick[:,:i+1] = 0
		err = error(x_ref, x_pick)
		mean[i] = np.mean(err)
		var[i] = np.std(err)

	print mean
	print var

def distance(e_low, e_high):
	return np.linalg.norm(e_low - e_high)

def eval_pattern_reconstruction(encoder, decoder, data_iterator):

	embedding = []
	x_data = []
	e_data = []
	h = 0
	for x, y in data_iterator:
		enc = encoder.predict(x)
		idx = np.random.choice(len(x), 10)
		e = encoder.predict(x[idx])
		if len(embedding) == 0:
			embedding = enc[:,-1]
			h = e.shape[1]
			x_data = x[idx]
			e_data = e
		else:
			embedding = np.concatenate((embedding, enc[:,-1]), axis=0)
			x_data = np.concatenate((x_data, x[idx]), axis=0)
			e_data = np.concatenate((e_data, e), axis=0)

	for idx, x in enumerate(x_data):
		for i in range(h):
			e_data[idx,i] = embedding[np.argmin([distance(e_data[idx,i], e) for e in embedding])]

	mean = np.zeros(h)
	var = np.zeros(h)
	for i in range(h):
		dec = decoder.predict(e_data[:,i])
		err = error(x_data, dec)
		mean[i] = np.mean(err)
		var[i] = np.std(err)

	print mean
	print var

def eval_nearest_neighbor(validation_data, training_data, n=100, n_input=15):
	error_score = [1000]*n
	error_x = [None]*n
	for i in tqdm(np.random.choice(len(training_data), n, replace=False)):
		for xs, _ in data_iterator:
			idx = np.random.choice(xs.shape[0], min(1000, xs.shape[0]), replace=False)
			for x in tqdm(xs[idx]):
				score = metrics.__pose_seq_error(x[:n_input], validation_data[i,:n_input])
				if score < error_score[i]:
					error_score[i] = score
					error_x[i] = np.copy(x[n_input:])

	error = [None]*n
	error_ = [None]*n
	for i in range(n):
		error[i] = metrics.__pose_seq_error(error_x[basename][i], gtp[basename][i], cumulative=True)
		# fk_animate.animate_compare(gt[basename][i], gtp[basename][i],
		# 	error_x[basename][i], 'Nearest Neighbor (1/%d)'%(DATA_ITER_SIZE/RANDOM_N),
		# 	pd[basename][i], 'Residual sup. (MA)', from_path+LOAD_PATH+'images/')

	_err = np.mean(error, axis=0)
	plt.plot(range(1,_err.shape[0]), _err)
	plt.xlabel('time-steps')
	plt.ylabel('error')
	plt.title('Nearest Neighbor (1/10)')
	plt.savefig('../results/nn-random-sampled-%d.png'%n)
	plt.close()


def test():
	y_true = np.reshape(np.arange(20), (5,2,2))
	y_predict = np.reshape(np.arange(20), (5,2,2))
	y_predict[0] = y_predict[0]+4
	y_predict[1] = y_predict[0]+2

	err = error(y_true, y_predict)
	print err

	mean = np.zeros(2)
	var = np.zeros(2)
	for i in range(2):
		err = error(y_true[:,:i+1], y_predict[:,:i+1])
		mean[i] = np.mean(err)
		var[i] = np.std(err)

	print mean
	print var

if __name__ == '__main__':
	test()