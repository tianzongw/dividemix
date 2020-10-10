import tensorflow as tf
from data.data_loader import dataset
from networks.network import get_model, train_model
from networks.utils import samplewise_loss, data2tensor, normlize_loss
from sklearn.mixture import GaussianMixture

threshold  = 0.01

cifar10 = dataset("cifar10", num_samples= 64)

print(cifar10.train_images.shape)

cifar10.noisify("symm", ratio= 0.4)


net1 =  get_model("preact", 32, 32, 3)
# net2 =  get_model("preact", 32, 32, 3)


# warmup
train_model(net1, cifar10.train_images, cifar10.train_labels, 8, 10)
cifar10_dataset = data2tensor(cifar10.train_images, cifar10.train_labels, 8)
net1_loss = samplewise_loss(net1, cifar10_dataset, all_metrics=False)
net1_loss = normlize_loss(net1_loss)



# model per-sample loss 
gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
gmm.fit(tf.reshape(net1_loss, (-1, 1)))
prob = gmm.predict_proba(tf.reshape(net1_loss, (-1, 1))) 
prob = prob[:,gmm.means_.argmin()]    
ind_labeled = prob > threshold
ind_unlabeled = prob < threshold

labeled_iamges, labeled_labels , unlabeled_images = cifar10.co_divide(ind_labeled, ind_unlabeled)
