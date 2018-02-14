import matplotlib.pyplot as plt
import numpy as np
from time import gmtime, strftime

from sklearn.decomposition import PCA as sklearnPCA

def see_embedding(encoder, data_iterator, args, concat=False):
		embedding = np.array([])
		for x, y in data_iterator:
			if concat:
				x = np.concatenate((x,y), axis=2)
			
			e = encoder.predict(x)

			if len(embedding) == 0:
				embedding = e
			else:
				embedding = np.concatenate((embedding, e), axis=0)

		plot(embedding, args)

def plot(embedding, args):
	pca = sklearnPCA(n_components=2) #2-dimensional PCA
	X_norm = (embedding - embedding.min())/(embedding.max() - embedding.min())
	transformed = pca.fit_transform(X_norm)

	plt.scatter(transformed[:,0], transformed[:,1], c='blue')
	
	for c in ['red', 'lightgreen', 'yellow']:
		n = np.random.randint(0, len(transformed)-10)
		random_sequence = transformed[n:n+10]	
		plt.scatter(random_sequence[:,0], random_sequence[:,1], c=c)
	# plt.scatter(lda_transformed[y==2][0], lda_transformed[y==2][1], label='Class 2', c='blue')
	# plt.scatter(lda_transformed[y==3][0], lda_transformed[y==3][1], label='Class 3', c='lightgreen')

	# Display legend and show plot
	# plt.legend(loc=3)
	plt.savefig('../out/embedding_'+ '-'.join(map(str, args)) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close()

	# plt.show()