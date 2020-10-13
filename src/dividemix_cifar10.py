import tensorflow as tf
from data.data_loader import dataset
from networks.network import get_model, train_model
from networks.utils import augment, samplewise_loss, data2tensor, normlize_loss
from sklearn.mixture import GaussianMixture

threshold  = 0.4
BATCH_SIZE = 1024
MAX_EPOCH = 10
cifar10 = dataset("cifar10")


cifar10.noisify("symm", ratio= 0.1)
cifar10_dataset = data2tensor(cifar10.train_images, cifar10.train_labels, BATCH_SIZE)


cifar10_augmented = augment(cifar10_dataset, BATCH_SIZE )
print("augmented")
net1 =  get_model("preact", 32, 32, 3)
# net2 =  get_model("preact", 32, 32, 3)


# # warmup
train_model(net1, cifar10_augmented,  BATCH_SIZE, MAX_EPOCH)




net1_loss = samplewise_loss(net1, cifar10_augmented, all_metrics=False)
net1_loss = normlize_loss(net1_loss)
# model per-sample loss 
gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
gmm.fit(tf.reshape(net1_loss, (-1, 1)))
prob = gmm.predict_proba(tf.reshape(net1_loss, (-1, 1))) 
prob = prob[:,gmm.means_.argmin()]    
ind_labeled = prob > threshold
ind_unlabeled = prob < threshold

labeled_images, labeled_labels , unlabeled_images, unlabeled_labels= cifar10.co_divide(ind_labeled, ind_unlabeled) # not augmented

labeled_dataset = data2tensor(labeled_images, labeled_labels, BATCH_SIZE)
unlabeled_dataset = data2tensor(unlabeled_images, unlabeled_labels, BATCH_SIZE)

labeled_dataset_augmented = augment(labeled_dataset, BATCH_SIZE)
unlabeled_dataset_augmented = augment(unlabeled_dataset, BATCH_SIZE)


# 

