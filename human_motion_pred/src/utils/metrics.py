import numpy as np
from tqdm import tqdm
import json
from scipy.spatial import distance

import translate__

DEFAULT_DIST = 1
OUT_DIR = '../new_out/'

def distance_name(mode):
	return ['l2', 'l1', 'cos', 'max', 'min'][mode]

def get_distance(e1, e2, mode=DEFAULT_DIST):
	if mode == 0:
		return np.linalg.norm(e1-e2)
	elif mode == 1:
		return np.sum(np.abs(e1-e2))
	elif mode == 2:
		return distance.cosine(e1,e2)
	elif mode == 3:
		return np.amax(np.abs(e1-e2))
	elif mode == 4:
		return np.amin(np.abs(e1-e2))

def __get_dist(embedding, z_ref, mode=DEFAULT_DIST):
	return [get_distance(embedding[i], z_ref, mode) for i in range(len(embedding))]

def __get_weights(embedding, z_ref, mode=DEFAULT_DIST):
	weights = __get_dist(embedding, z_ref, mode)
	w_i = np.argsort(weights)
	return weights, w_i

# Pattern matching methods

def __closest(embedding, z_ref, weights=[], return_weight=False):
	if not any(weights):
		weights = __get_dist(embedding, z_ref)
	idx = np.argmin(weights)
	if return_weight:
		return embedding[idx], weights[idx]
	return embedding[idx]

def __normalized_distance_mean(embedding, weights, w_i):
	if weights[w_i[0]] < 1e-16 or weights[w_i[0]] == float('-inf'):
		return embedding[w_i[0]]
	return np.sum([embedding[d]/weights[d] for d in w_i], axis=0)/np.sum([1.0/weights[d] for d in w_i])

def __mean(embedding, z_ref, n=45, weights=[], w_i=[]):
	if not any(weights):
		weights, w_i = __get_weights(embedding, z_ref)
	if n > 0:
		w_i = w_i[:n]
	return __normalized_distance_mean(embedding, weights, w_i)

def __random(embedding, z_ref, n=-1, weights=[], w_i=[]):
	if n > 0:
		if not any(weights):
			embedding = embedding[np.random.choice(embedding.shape[0], n, replace=False)]
		else:
			w_idx = sorted(np.random.choice(len(w_i), n, replace=False).tolist())
			w_i = [w_i[i] for i in w_idx]

	return __mean(embedding, z_ref, weights=weights, w_i=w_i)

def __closest_partial_index(embedding_partial, z_ref, weights={}):
	if not any(weights):
		weights = __get_dist(embedding_partial, z_ref)
	return np.argmin(weights)

def add_get_params(embedding):
	diff = embedding[:,-1] - embedding[:,0]
	return np.mean(diff, axis=0), np.std(diff, axis=0)

# load embedding:

def load_embedding(model, data_gen, subspace=-1):
	embedding = []
	for k, x in data_gen:
		e = model.encoder.predict(x)
		if subspace > 0:
			e = e[:, subspace]
		else:
			e = e[:, model.hierarchies]
		if len(embedding) == 0:
			embedding = e
		else:
			embedding = np.concatenate((embedding, e), axis=0)
		# break
	print 'embedding shape', embedding.shape
	return embedding


def get_mesures(model, embedding, test_x, test_y):
	action_n = len(model.labels)
	sample_n = 8
	pred_t = model.timesteps - model.conditioned_pred_steps

	input_x = np.zeros(N, model.timesteps, model.data_dim)
	input_x[:,:model.timesteps-model.conditioned_pred_steps] = test_x[:,-model.conditioned_pred_steps:]
	encoded = model.encoder.predict(input_x)[:,model.conditioned_pred_steps-1]
	encoded = np.reshape(encoded, (action_n, sample_n, encoded.shape[-1]))
	test_y = np.reshape(test_y[:,:pred_t], (action_n, sample_n, pred_t, test_y.shape[-1]))

	methods = ['mean', 'closest', 'partial_closest', 'add']
	add_mean, add_std = add_get_params(embedding)
	print 'add std', np.mean(add_std)

	scores = {k: np.zeros(action_n, sample_n, pred_t) for k in methods}
	for i in range(action_n):
		for j in range(sample_n):
			z_ref = encoded[i, j]
			ls = embedding[:,1]
			weights, w_i = __get_weights(ls, z_ref)
			zs = np.zeros((len(methods), z_ref.shape[-1]))

			for method in methods:
				if method == 'mean':
					new_e = __mean(ls, z_ref, 30, weights, w_i)
				elif method == 'closest':
					new_e = __closest(ls, z_ref, weights)
				elif method == 'partial_closest':
					new_e_idx = __closest_partial_index(embedding[:,0], z_ref)
					new_e = ls[new_e_idx]
				else:
					new_e = z_ref + add_mean

				zs[i] = new_e

			pred_x = model.decoder.predict(new_e)
			gt = np.tile(y_test[i,j], [len(methods), 1, 1])
			err = translate__.euler_diff(gt, pred_x, model)[1]
			for k, method in enumerate(methods):
				scores[method][i,j] = err[k]

	for method in methods:
		print method
		for i, action in enumerate(model.labels):
			print action, np.mean(scores[method][i], axis=0)
		print 'total', np.mean(np.mean(scores[method], axis=0), axis=0)
		scores[method] = scores[method].tolist()

	with open(OUT_DIR+'get_mesures_%d'%(model.model_signature)+__get_timestamp()+'.json', 'w') as jsonfile:
		json.dump(scores, jsonfile)
