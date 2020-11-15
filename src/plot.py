import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == '__main__':

	root_dir = '../logs'
	noise = ['0.2', '0.5', '0.8', '0.9']
	epochs = {}
	accs = {}
	best_acc = {}
	last_acc = {}
	epochs_stats = {}
	aucs = {}
	labeled_samples = {}

	for (dirpath, dirnames, filenames) in os.walk(root_dir):
		for file in filenames:
			if 'txt' in file:
				if 'acc' in file:
					epoch = []
					acc = []	
					with open(os.path.join(dirpath, file)) as fp:
						for _, line in enumerate(fp):
							epoch.append(int(line.split(' ')[0].split(':')[-1]))
							acc.append(float(line.split(' ')[-1].split(':')[-1])/100)
					
					epochs[file] = epoch
					accs[file] = acc
					best_acc[file] = np.max(acc)
					last_acc[file] = acc[-1]

				elif 'stat' in file:	
					epoch = []
					n_labeld_samples = []
					auc = []
					with open(os.path.join(dirpath, file)) as fp:
						for cnt, line in enumerate(fp):
							if cnt % 2 == 0:
								epoch.append(cnt/2)
								n_labeld_samples.append(int(line.split(':')[1].split(' ')[0]))
								auc.append(float(line.split(':')[-1]))
					
					epochs_stats[file] = epoch
					labeled_samples[file] = n_labeld_samples
					aucs[file] = auc

	# for key in best_acc:
	# 	print('best', key, best_acc[key])
	# 	print('last', key, last_acc[key])
	
		for key, value in epochs_stats.items():	
			if '0.5' in key:
				if 'asym' in key:
					label = key[-14:-4] if 'ablation' in key else 'original'
					plt.plot(epochs_stats[key], labeled_samples[key], label = label)
		
		plt.xlabel('epoch')
		plt.ylabel('number of cleaned samples')
		plt.legend(loc = 1)
		fig_path = os.path.join(root_dir, 'plots', '0.5_asym_labeled_samples.jpg')
		plt.savefig(fig_path)
		plt.clf()


		for r in noise:	
			for key, value in epochs_stats.items():
				if not 'asym' in key:
					if r in key:
						label = key[-14:-4] if 'ablation' in key else 'original'
						plt.plot(epochs_stats[key], labeled_samples[key], label = label)

			plt.xlabel('epoch')
			plt.ylabel('number of cleaned samples')
			plt.legend(loc = 1)
			fig_path = os.path.join(root_dir, 'plots', r + '_sym_labeled_samples.jpg')
			plt.savefig(fig_path)
			plt.clf()


		for key, value in epochs_stats.items():	
			if '0.5' in key:
				if 'asym' in key:
					label = key[-14:-4] if 'ablation' in key else 'original'
					plt.plot(epochs_stats[key], aucs[key], label = label)
		
		plt.xlabel('epoch')
		plt.ylabel('AUC of cleaned samples')
		plt.legend(loc = 1)
		fig_path = os.path.join(root_dir, 'plots', '0.5_asym_AUC.jpg')
		plt.savefig(fig_path)
		plt.clf()


		for r in noise:	
			for key, value in epochs_stats.items():
				if not 'asym' in key:
					if r in key:
						label = key[-14:-4] if 'ablation' in key else 'original'
						plt.plot(epochs_stats[key], aucs[key], label = label)

			plt.xlabel('epoch')
			plt.ylabel('AUC of cleaned samples')
			plt.legend(loc = 1)
			fig_path = os.path.join(root_dir, 'plots', r + '_sym_AUC.jpg')
			plt.savefig(fig_path)
			plt.clf()


		for key, value in epochs.items():
			if '0.5' in key:
				if 'asym' in key:
					label = key[-14:-4] if 'ablation' in key else 'original'
					plt.plot(epochs[key], accs[key], label = label)
				
		plt.xlabel('epochs')
		plt.ylabel('test accuracy')
		plt.legend()
		plt.savefig(os.path.join(dirpath, 'plots', '0.5_asym_acc.jpg'))
		plt.clf()
		
		
		for r in noise:
			for key, value in epochs.items():
				if not 'asym' in key:
					if r in key:
						label = key[-14:-4] if 'ablation' in key else 'original'
						plt.plot(epochs[key], accs[key], label = label)
						
					
			plt.xlabel('epochs')
			plt.ylabel('test accuracy')
			plt.legend()
			plt.savefig(os.path.join(dirpath, 'plots', r + '_sym_acc.jpg'))
			plt.clf()

			
			