import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import gmtime, strftime
import json

H_LSTM = 0
ML_LSTM = 1

CLOSEST = 0
MEAN = 1
RANDOM = 2


OUT_DIR = '../out/'

def __get_timestamp():
	return strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime())

def __plot(x, ys, errs, labels, x_label, y_label, x_ticks, title, model_name, baseline=None):
	for i in range(len(ys)):
		plt.plot(x, ys[i], label=labels[i])
		plt.fill_between(x, ys[i]-errs[i], ys[i]+errs[i], alpha=0.3)
	if baseline:
		y = np.array([baseline[0] for _ in range(len(x))])
		err = np.array([baseline[1] for _ in range(len(x))])
		plt.plot(x, y, label='Baseline', color='#999999')
		plt.fill_between(x, y-err, y+err, label='Baseline', facecolor='#999999', alpha=0.3)
	plt.xticks(x, x_ticks)
	plt.legend()
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	# plt.show()
	plt.savefig(OUT_DIR+'%s_%s'%(title, model_name) + __get_timestamp() + '.png') 
	plt.close()

def __get_dist(embedding, z_ref):
	return [np.linalg.norm(embedding[i]-z_ref) for i in range(len(embedding))]

def __get_weights(embedding, z_ref):
	weights = __get_dist(embedding, z_ref)
	w_i = np.argsort(weights)
	return weights, w_i

def __closest(embedding, z_ref, weights=[]):
	if not any(weights):
		weights = __get_dist(embedding, z_ref)
	return embedding[np.argmin(weights)]

def __mean(embedding, z_ref, n=30, weights=[], w_i=[]):
	if not any(weights):
		weights, w_i = __get_weights(embedding, z_ref)
	if n > 0:
		w_i = w_i[:n]
	if weights[w_i[0]] < 1e-16 or weights[w_i[0]] == float('-inf'):
		return embedding[w_i[0]]
	return np.sum([embedding[d]/weights[d] for d in w_i], axis=0)/np.sum([1.0/weights[d] for d in w_i])

def __random(embedding, z_ref, n=-1, weights=[], w_i=[]):
	if n > 0:
		if not any(weights):
			embedding = embedding[np.random.choice(embedding.shape[0], n, replace=False)]
		else:
			w_idx = sorted(np.random.choice(len(w_i), n, replace=False).tolist())
			w_i = [w_i[i] for i in w_idx]

	return __mean(embedding, z_ref, weights=weights, w_i=w_i)

def __get_latent_reps(encoder, pose, model_name, n=-1):
	if model_name == H_LSTM:
		enc = encoder.predict(pose)
		if n > 0:
			return enc[:,n]
		return enc
	else:
		b, h, d = pose.shape
		x = None
		if n > 0:
			x = np.zeros((b,h,d))
			x[:,:n+1] = pose[:,:n+1]
			return encoder.predict(x)
		else:
			x = np.zeros((b*h,h,d))
			for i in range(b):
				for j in range(h):
					x[i*h+j,:j+1] = pose[i,:j+1]
			return np.reshape(encoder.predict(x), (b, h, -1))

def __get_subspace(embedding, n, model_name):
	if model_name == H_LSTM:
		return embedding[:,n]
	else:
		return embedding[n]

def __cuts():
	return np.array([0,2,4,7,9])
	# return [4]

def __rn():
	# return [100, 200, -1, -2]
	return [100, 500, 1000, 5000, 10000, 50000, 55000 -1, -2]

def __common_params(n):
	cuts = __cuts()
	rn =__rn()
	scores = {'score':{r:{c:[0]*n for c in cuts} for r in rn}, 
			'e_score':{r:{c:[0]*n for c in cuts} for r in rn}}
	return cuts, rn, scores

def __latent_error(z_ref, z_pred):
	return np.linalg.norm(z_ref - z_pred)

def __pose_error(pose_ref, pose_pred):
	return np.mean([np.linalg.norm(pose_ref[t]-pose_pred[t]) for t in range(len(pose_pred))])

def random_baseline(validation_data):
	n = 100
	x_ref = validation_data[np.random.choice(len(validation_data), n)]
	x_pick = validation_data[np.random.choice(len(validation_data), n)]
	h = x_ref.shape[1]

	mean = np.zeros(h)
	var = np.zeros(h)
	for i in range(h):
		err = [__pose_error(x_ref[j,i:], x_pick[j,i:]) for j in range(n)]
		mean[i] = np.mean(err)
		var[i] = np.std(err)

	print mean
	print var

def validate(validation_data, encoder, decoder, h, model_name):
	mean = np.zeros(h)
	std = np.zeros(h)
	mean_sub = np.zeros(h)
	std_sub = np.zeros(h)
	sample_n = validation_data.shape[0]
	poses = np.zeros(validation_data.shape)
	enc = __get_latent_reps(encoder, validation_data, model_name)
	for i in range(h):
		poses[:,:i+1] = validation_data[:,:i+1]
		p_poses = decoder.predict(enc[:,i])
		err = [__pose_error(poses[j], p_poses[j]) for j in range(sample_n)]
		mean[i] = np.mean(err)
		std[i] = np.std(err)
		err = [__pose_error(poses[j,:i+1], p_poses[j,:i+1]) for j in range(sample_n)]
		mean_sub[i] = np.mean(err)
		std_sub[i] = np.std(err)

	def __r(a):
		return np.around(a*100, 3).tolist()

	print sample_n
	print __r(mean)
	print __r(std)
	print __r(np.mean(mean))
	print __r(np.sqrt(np.sum([s**2 for s in std])/sample_n))
	print __r(mean_sub)
	print __r(std_sub)
	print __r(np.mean(mean_sub))
	print __r(np.sqrt(np.sum([s**2 for s in std_sub])/sample_n))

def __pca_reduce(embedding):
	from sklearn.decomposition import PCA as sklearnPCA
	pca = sklearnPCA(n_components=2) #2-dimensional PCA
	X_norm = (embedding - embedding.min())/(embedding.max() - embedding.min())
	transformed = pca.fit_transform(X_norm)
	return transformed	

def plot_convex_hall(embedding, h, model_name):
	from scipy.spatial import ConvexHull
	sample_n = embedding.shape[0] if model_name == H_LSTM else embedding.shape[1]
	e_list = np.concatenate(embedding, axis=0)
	transformed = __pca_reduce(e_list)
	for i in range(h):
		choices = transformed[i::h] if model_name == H_LSTM else transformed[i*sample_n:(i+1)*sample_n]
		hull = ConvexHull(choices)
		for simplex in hull.simplices:
			plt.plot(choices[simplex, 0], choices[simplex, 1], c='#'+('%s'%('{0:01X}'.format(i+2)))*6)
		# plt.scatter(choices[:,0], choices[:,1], c='#'+('%s'%('{0:01X}'.format(i+2)))*6)
	
	for c in ['red', 'green', 'blue']:
		n = np.random.randint(0, sample_n-50)
		for i, s in enumerate(['x', '+', 'd']):
			random_sequence = (transformed[i*(h-1)/2::h])[n:n+50:10] if model_name == H_LSTM else (transformed[i*(h-1)/2*sample_n:])[n:n+50:10]
			plt.scatter(random_sequence[:,0], random_sequence[:,1], c=c, marker=s)

	# plt.legend(loc=3)
	plt.savefig(OUT_DIR+'plot_convex_halls_%s'%(model_name) + strftime("%a-%d-%b-%Y-%H_%M_%S", gmtime()) + '.png') 
	plt.close()
	# plt.show()

def plot_metrics(embedding, validation_data, encoder, decoder, h, model_name, n_valid = 100):
	cuts, rn, scores = __common_params(n_valid)
	given_n = h/2-1
	new_e = None

	idxs = np.random.choice(len(validation_data), n_valid)
	enc = __get_latent_reps(encoder, validation_data[idxs], model_name)
	for k, idx in enumerate(tqdm(idxs)):
		pose = np.copy(validation_data[idx])
		z_ref = enc[k,given_n]
		for cut in cuts:
			ls = __get_subspace(embedding, cut, model_name)
			weights, w_i = __get_weights(ls, z_ref)
			zs = np.zeros((len(rn), z_ref.shape[-1]))
			for i,n in enumerate(rn):
				if n > -2:
					new_e = __random(ls, z_ref, n, weights, w_i)
				else:
					new_e = __closest(ls, z_ref, weights)

				scores['e_score'][n][cut][k] = __latent_error(enc[k,cut], new_e)
				zs[i] = new_e
			p_poses = decoder.predict(np.array(zs))
			t_pose = np.zeros(pose.shape)
			t_pose[:cut+1] = pose[:cut+1]
			for i, n in enumerate(rn):
				scores['score'][n][cut][k] = np.mean([np.linalg.norm(t_pose[t]-p_poses[i,t]) for t in range(h)])

	x = np.arange(len(rn))
	ys = np.array([[np.mean(scores['score'][n][cut]) for n in rn] for cut in cuts])
	errs = np.array([[np.std(scores['score'][n][cut]) for n in rn] for cut in cuts])
	labels = map(str, cuts+1)
	x_ticks = map(str, rn[:-2])+['all', 'all (CLOSEST)']
	x_label = 'Number of random latent representation'
	y_label = 'Error'
	title = 'Pose error'
	__plot(x, ys, errs, labels, x_label, y_label, x_ticks, title, model_name)
	
	ys_e = np.array([[np.mean(scores['e_score'][n][cut]) for n in rn] for cut in cuts])
	errs_e = np.array([[np.std(scores['e_score'][n][cut]) for n in rn] for cut in cuts])
	title = 'Latent error'
	__plot(x, ys_e, errs_e, labels, x_label, y_label, x_ticks, title, model_name)

	with open(OUT_DIR+'plot_metrics_%d'%(model_name)+__get_timestamp()+'.json', 'w') as jsonfile:
		json.dump({
			'x':x.tolist(), 
			'score':ys.tolist(), 
			'score_std':errs.tolist(),
			'e_score':ys_e.tolist(),
			'e_score_std':errs_e.tolist()
			}, jsonfile)

# def load_and_plot_metrics():
# 	labels = map(str, __cuts()+1)
# 	x_ticks = map(str, __rn()[:-2])+['all', 'all (CLOSEST)']
# 	r_mean = np.mean([1.16885212, 1.16788286, 1.16698216, 1.16632305, 1.1657963, 1.16548061, 1.1655127, 1.16632586, 1.16714065, 1.16816711])
# 	r_std = np.mean([0.50485349, 0.50628007, 0.50789828, 0.50992223, 0.51254757, 0.51551781, 0.51899528, 0.522206, 0.52444423, 0.52576033])

# 	x =  np.array([0, 1, 2, 3, 4, 5, 6, 7])
# 	x_label = 'Number of random latent representation'
# 	y_label = 'Error'
# 	title = 'Pose error'
# 	ys_e = np.array([[0.12038144839939598, 0.12438103523433958, 0.14027112580314183, 0.15099315101257052, 0.18120085638468938, 0.1902318145641235, 0.19687644564498905, 0.21320699162346793], [0.15521563585865053, 0.1470137967968643, 0.15474798572781936, 0.13410950303022437, 0.12333566789509348, 0.11405399243008908, 0.11977066369117302, 0.1535186861463435], [0.22385166854492708, 0.17862170076411465, 0.14089239019554703, 0.11588302654113676, 0.10195729822726479, 0.08722791321051038, 0.08164474928130484, 0.08087280956037424], [0.40010332663659337, 0.3057915251496957, 0.24976986089611544, 0.22543247695747826, 0.19750538421663122, 0.12513479921791867, 0.12764647479212435, 0.11234540015190575], [0.5096788517901784, 0.36355400391139314, 0.3409341041185303, 0.29627260491972235, 0.23186480685187225, 0.19839312482482177, 0.19607627462878985, 0.2061184260622021]])
# 	errs_e =  np.array([[0.01300764760878271, 0.014022754573477805, 0.012625930741606237, 0.02791979828255813, 0.031432166669020864, 0.052014328849237235, 0.053675775997156464, 0.048246118028158556], [0.020166207474957263, 0.01602045433625464, 0.014503441057039658, 0.02078702274688898, 0.007246900132830925, 0.013278838856637646, 0.018078766440430168, 0.08407078007707729], [0.04608033181134376, 0.04179752164961082, 0.02470869875322068, 0.035737383834718014, 0.028223244466255765, 0.026834188481573283, 0.028352556530983487, 0.02729040677140092], [0.1620494404038322, 0.11385758406917847, 0.059851528921452075, 0.058553154929113435, 0.03615444619705744, 0.0438890622694841, 0.041157891190979144, 0.022881201379869102], [0.16959154806097107, 0.10225854668564115, 0.09036991605882948, 0.06286424694431576, 0.06988227664736507, 0.07614003192884179, 0.0773631658725451, 0.10912618704303215]])
# 	__plot(x, ys_e, errs_e, labels, x_label, y_label, x_ticks, title, ML_LSTM, (r_mean, r_std))
# 	ys_e =  np.array([[0.20137719810009003, 0.19854135811328888, 0.23294483125209808, 0.25528615713119507, 0.30040252208709717, 0.32202818989753723, 0.3338886797428131, 0.37445178627967834], [0.15560470521450043, 0.13506750762462616, 0.12827028334140778, 0.10386895388364792, 0.08855707198381424, 0.0810825377702713, 0.08436111360788345, 0.10369466990232468], [0.1480947732925415, 0.12370779365301132, 0.0905865952372551, 0.05210699141025543, 0.02919320948421955, 0.019521795213222504, 0.005039839539676905, 0.0], [0.19472980499267578, 0.17257004976272583, 0.1461387425661087, 0.11877650022506714, 0.09354287385940552, 0.050211403518915176, 0.05202200636267662, 0.040467917919158936], [0.24467794597148895, 0.2189996838569641, 0.2054140567779541, 0.18302279710769653, 0.1324988454580307, 0.12119889259338379, 0.12161127477884293, 0.13099737465381622]])
# 	errs_e =  np.array([[0.019996805116534233, 0.03797980770468712, 0.04235927388072014, 0.062485214322805405, 0.05214828997850418, 0.08480076491832733, 0.08496524393558502, 0.059506773948669434], [0.05304839462041855, 0.03516553342342377, 0.02236180193722248, 0.021917888894677162, 0.010506452061235905, 0.014312404207885265, 0.022116975858807564, 0.0770457461476326], [0.03898977115750313, 0.050432123243808746, 0.02627512812614441, 0.037525590509176254, 0.031086161732673645, 0.018626779317855835, 0.007127409800887108, 0.0], [0.07278334349393845, 0.07068289816379547, 0.04892962425947189, 0.03601483255624771, 0.02791794203221798, 0.03510011360049248, 0.034834232181310654, 0.005154438782483339], [0.07135862112045288, 0.07195665687322617, 0.0728001818060875, 0.059160832315683365, 0.0641264021396637, 0.06564708054065704, 0.06755100190639496, 0.07556063681840897]])
# 	__plot(x, ys_e, errs_e, labels, x_label, y_label, x_ticks, title, H_LSTM, (r_mean, r_std))

# 	title = 'Latent error'
# 	ys_e =  np.array([[0.10340486589082926, 0.09507400554441556, 0.09326923338246867, 0.09065486722707933, 0.09058566934444495, 0.09233659594018218, 0.09165474348598945, 0.09728895613005797], [0.1998628817833906, 0.15680082349635838, 0.14578566314662728, 0.12810276831897002, 0.12174802781423048, 0.11343847847699173, 0.11282092025254789, 0.11813634040518116], [0.28960668399033107, 0.20677842428341373, 0.18086588686325442, 0.13434860711985994, 0.12080852905768591, 0.08703466444099149, 0.08614546746941426, 0.08067741847933439], [0.4443636313341576, 0.3293818153912774, 0.29207228528764495, 0.22737887454527617, 0.20096607153602686, 0.15588546416224985, 0.15375246596217693, 0.1108654160117954], [0.5603737362378314, 0.4193303536315243, 0.3695692646792849, 0.2878249768768396, 0.25751726718987117, 0.20046016790497884, 0.1976279424471294, 0.18107765341170484]])
# 	errs_e =  np.array([[0.0261484309680495, 0.017805190500638446, 0.017180162903567664, 0.02008374570781163, 0.02186868976845766, 0.022430051115878123, 0.021843506850267987, 0.02302749599915059], [0.07075366832968362, 0.050441027757330434, 0.045560981702542976, 0.03729550318714312, 0.03116220358515678, 0.027446440312838886, 0.026473311957709585, 0.03476537089335328], [0.12017743046727375, 0.07958684958302575, 0.07214862874938523, 0.053388099182942135, 0.045533060273864195, 0.030086939672544208, 0.029653227575883522, 0.028319526875941713], [0.17187992269172525, 0.13064333090511154, 0.1082422825883021, 0.08648614352763734, 0.07526716762636701, 0.05951128873074975, 0.05989073026579775, 0.04536635351142736], [0.23685200825891112, 0.16090971895198522, 0.13996158984069756, 0.1084458037115277, 0.1002440431707891, 0.09536947739533398, 0.09298122644339035, 0.11328031262578839]])
# 	__plot(x, ys_e, errs_e, labels, x_label, y_label, x_ticks, title, H_LSTM, (r_mean, r_std))
# 	ys_e =  np.array([[0.2172783464193344, 0.18302960693836212, 0.16993457078933716, 0.13770873844623566, 0.1275586485862732, 0.11440835148096085, 0.11117168515920639, 0.12206842750310898], [0.21430246531963348, 0.1711989790201187, 0.15671242773532867, 0.12784355878829956, 0.11356261372566223, 0.09424544125795364, 0.09293364733457565, 0.09365426748991013], [0.21834787726402283, 0.16394329071044922, 0.13843360543251038, 0.08964794874191284, 0.07148588448762894, 0.013829534873366356, 0.013654354028403759, 8.928331851620896e-09], [0.24211663007736206, 0.19903972744941711, 0.18094392120838165, 0.14203011989593506, 0.1255008727312088, 0.08772356063127518, 0.08548028022050858, 0.03610461577773094], [0.2717079818248749, 0.22966201603412628, 0.21287836134433746, 0.17392224073410034, 0.15870435535907745, 0.1235559731721878, 0.12138669937849045, 0.10411210358142853]])
# 	errs_e =  np.array([[0.09207379817962646, 0.06566637754440308, 0.05729806050658226, 0.04544539377093315, 0.04853569716215134, 0.05433185398578644, 0.05302739515900612, 0.06853926181793213], [0.08678069710731506, 0.06983590126037598, 0.06204288825392723, 0.048173706978559494, 0.03934480622410774, 0.040443357080221176, 0.038727350533008575, 0.06305296719074249], [0.0887117013335228, 0.06942052394151688, 0.068161740899086, 0.05411987006664276, 0.04646912217140198, 0.02678956463932991, 0.02398272603750229, 4.385157126307604e-08], [0.08421404659748077, 0.07475294172763824, 0.06613551825284958, 0.05545832961797714, 0.05095996707677841, 0.04531259089708328, 0.04468590393662453, 0.03698369488120079], [0.09128232300281525, 0.07770057022571564, 0.07284701615571976, 0.06611369550228119, 0.0650617778301239, 0.0688294991850853, 0.06638599187135696, 0.0814661756157875]])
# 	__plot(x, ys_e, errs_e, labels, x_label, y_label, x_ticks, title, H_LSTM, (r_mean, r_std))


def gen_random(embedding, validation_data, encoder, decoder, h, model_name, numb=10):
	import image
	idxs = np.random.randint(0, len(validation_data), numb)
	enc = __get_latent_reps(encoder, validation_data[idxs], model_name)
	for i, n in enumerate(tqdm(idxs)):
		pose = np.copy(validation_data[n])
		pose[h/2:] = 0
		z_ref = enc[i,h/2-1]
		zs = []
		# cut variation
		for cut in __cuts():
			ls = __get_subspace(embedding, cut, model_name)
			new_e = __random(ls, z_ref, 10000)
			zs.append(new_e)
		p_poses = decoder.predict(np.array(zs))
		image.plot_poses([pose, validation_data[n]], p_poses, title='Pattern matching (random) (prediction in bold)')

		zs = []
		ls = __get_subspace(embedding, h-1, model_name)
		weights, w_i = __get_weights(ls, z_ref)
		# random variation
		for nn in __rn()[:-1]:
			new_e = __random(ls, z_ref, nn, weights, w_i)
			zs.append(new_e)

		p_poses = decoder.predict(np.array(zs))
		image.plot_poses([pose, validation_data[n]], p_poses, title='Pattern matching (random) (prediction in bold)')

def go_up_hierarchy(embedding, validation_data, encoder, decoder, h, model_name, cut=1, l_n=10, numb=10, method=CLOSEST, nn=10000):
	import image
	idxs = np.random.randint(0, len(validation_data)-l_n, numb)
	for i, n in enumerate(tqdm(idxs)):
		enc = __get_latent_reps(encoder, np.array(validation_data[n:n+1]), model_name)
		pose = np.copy(validation_data[n])
		pose[cut+1:] = 0
		z_ref = enc[0,cut]
		zs = [z_ref]
		for i in range(1,h,2):
			e = __get_subspace(embedding, i, model_name)
			new_e = None
			if method == CLOSEST:
				new_e = __closest(e, z_ref)
			elif method == MEAN:
				new_e = __mean(e, z_ref)
			else:
				new_e = __random(e, z_ref, nn)
			zs.append(new_e)
		p_poses = decoder.predict(np.array(zs))
		image.plot_poses([pose, validation_data[n]], p_poses, title='Pattern matching (best) (prediction in bold)')

def gen_long_sequence(embedding, validation_data, encoder, decoder, h, model_name, l_n=25, numb=10, nn=-1):
	import image
	idxs = np.random.randint(0, len(validation_data)-l_n, numb)
	ls = __get_subspace(embedding, h-1, model_name)
	err = np.zeros(numb)
	for i, n in enumerate(tqdm(idxs)):
		true_pose = np.reshape(validation_data[n:n+l_n,0], (1, l_n, -1))
		# true_pose = validation_data[n:n+l_n]
		poses = np.zeros((l_n/(h/2), l_n, true_pose.shape[-1]))
		poses[0, :h] = validation_data[n]
		current_pose = validation_data[n:n+1]
		for j in range(l_n/(h/2)-2):
			pose = np.zeros(current_pose.shape)
			pose[0,:h/2] = current_pose[0,h/2:]
			z_ref = __get_latent_reps(encoder, pose, model_name, h/2-1)
			new_e = __random(ls, z_ref, nn)
			current_pose = decoder.predict(np.array([new_e]))
			poses[j+1,(j+1)*(h/2):(j+3)*(h/2)] = np.copy(current_pose[0,])
			# poses[j,j+1:j+1+h] = current_pose[0]

		poses[-1] = np.sum(poses, axis=0)
		poses[-1, h/2:l_n-h/2] = poses[-1, h/2:l_n-h/2]/2
		err[i] = __pose_error(true_pose[0], poses[-1])
		image.plot_poses(true_pose, poses, title='Pattern matching (long) (prediction in bold)')

	print np.mean(err), np.std(err)