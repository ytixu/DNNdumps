import glob
import matplotlib.pyplot as plt

def get_logs(files_exp):
	data = {}
	for logfile_name in glob.glob(files_exp):
		with open(logfile_name, 'r') as logfile:
			for line in logfile.readlines():
				log = line.split(' :: ')
				data[log[0]] = map(float, log[1].split(' '))

	return data

def plot(data):
	for name, seq in data.iteritems():
		time = range(len(seq))
		plt.plot(time, seq)

	plt.show()

plot(get_logs('../lstm_ae_losses.txt'))