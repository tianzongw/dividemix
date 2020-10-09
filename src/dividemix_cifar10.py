import tensorflow as tf
from data.data_loader import dataset
from networks.network import get_model, train_model
from networks.utils import samplewise_loss, data2tensor
from sklearn.mixture import GaussianMixture


cifar10 = dataset("cifar10")

cifar10.noisify("symm", ratio= 0.4)


net1 =  get_model("preact", 32, 32, 3)
# net2 =  get_model("preact", 32, 32, 3)


# warmup
train_model(net1, cifar10.train_images, cifar10.train_labels, 8, 10)
cifar10_dataset = data2tensor(cifar10.train_images, cifar10.train_labels, 8)
net1_loss = samplewise_loss(net1, cifar10_dataset, all_metrics=False)

# model per-sample loss 
gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
gmm.fit(tf.reshape(net1_loss, (-1, 1)))
prob = gmm.predict_proba(tf.reshape(net1_loss, (-1, 1))) 
prob = prob[:,gmm.means_.argmin()]    
