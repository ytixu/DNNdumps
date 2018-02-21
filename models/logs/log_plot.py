import glob
import matplotlib.pyplot as plt
import numpy as np

AVG_N = 10

def get_logs(files_exp):
	data = {}
	for logfile_name in glob.glob(files_exp):
		with open(logfile_name, 'r') as logfile:
			for line in logfile.readlines():
				log = line.split(' :: ')
				data[log[0]] = np.array(map(float, log[1].split(' ')))
				data[log[0]] = np.array([np.mean(data[log[0]][i:i+AVG_N]) for i in range(len(data[log[0]]-AVG_N+1))])
				data[log[0]] = np.array([np.mean(data[log[0]][i:i+AVG_N]) for i in range(len(data[log[0]]-AVG_N+1))])
				data[log[0]] = np.array([np.mean(data[log[0]][i:i+AVG_N]) for i in range(len(data[log[0]]-AVG_N+1))])
				data[log[0]] = [np.mean(data[log[0]][i:i+AVG_N]) for i in range(len(data[log[0]]-AVG_N+1))]

	return data

def plot(data):
	for name, seq in data.iteritems():
		time = range(len(seq))
		plt.plot(time, seq, label=name)

	plt.legend()
	plt.show()

plot(get_logs('../*.log'))