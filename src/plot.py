import matplotlib.pyplot as plt
import os


if __name__ == '__main__':

	root_dir = '../logs'
	for (dirpath, dirnames, filenames) in os.walk(root_dir):
		for file in filenames:
			if 'txt' in file:
			
				if 'acc' in file:
					epochs = []
					accs = []
					with open(os.path.join(dirpath, file)) as fp:
						for _, line in enumerate(fp):
							epochs.append(int(line.split(' ')[0].split(':')[-1]))
							accs.append(float(line.split(' ')[-1].split(':')[-1])/100)
					
				
					plt.plot(epochs, accs)
					plt.xlabel('epochs')
					plt.ylabel('test accuracy')
					
					fig_path = os.path.join(root_dir, 'plots', file[8:-4]+'.jpg')
					plt.savefig(fig_path)
				
				if 'stat' in file:
					epochs = []
					n_labeld_samples = []
					auc = []
					with open(os.path.join(dirpath, file)) as fp:
						for cnt, line in enumerate(fp):
							if cnt % 2 == 0:
								epochs.append(cnt/2)
								n_labeld_samples.append(int(line.split(':')[1].split(' ')[0]))
								auc.append(float(line.split(':')[-1]))
					
					fig, axs = plt.subplots(1,2)
					axs[0].plot(epochs, n_labeld_samples)
					axs[0].set_xlabel('epoch')
					axs[0].set_ylabel('number of labeled samples')

					axs[1].plot(epochs, auc)
					axs[1].set_xlabel('epoch')
					axs[1].set_ylabel('AUC for cleaned samples')
					
					fig_path = os.path.join(root_dir, 'plots', file[8:-4]+'.jpg')
					plt.savefig(fig_path)
