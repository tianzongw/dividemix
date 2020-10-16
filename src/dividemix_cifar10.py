import tensorflow as tf
import numpy as np
from data.data_loader import dataset
from networks.network import get_model, train_model
from networks.utils import augment, samplewise_loss, data2tensor, normlize_loss, predict_batchwise, sharpen
from sklearn.mixture import GaussianMixture

threshold  = 0.4
BATCH_SIZE = 1024
WARMUP_EPOCH = 1
MAX_EPOCH = 2
W_P = 0.5
T = 0.5
M = 2
cifar10 = dataset("cifar10")


cifar10.noisify("symm", ratio= 0.1)
cifar10_dataset = data2tensor(cifar10.train_images, cifar10.train_labels, BATCH_SIZE)


cifar10_augmented = augment(cifar10_dataset, BATCH_SIZE)
net1 =  get_model("preact", 32, 32, 3)

# net2 = net1
# net2 =  get_model("preact", 32, 32, 3)


# warmup
try: 
    net1.load_weights('./models/checkpoint')
except:
    train_model(net1, cifar10_augmented,  BATCH_SIZE, WARMUP_EPOCH)
    net1.save_weights('./models/checkpoint')

# for epoch in MAX_EPOCH:



# model per-sample loss 
net1_loss = samplewise_loss(net1, cifar10_augmented, all_metrics=False)
net1_loss = normlize_loss(net1_loss)
gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
gmm.fit(tf.reshape(net1_loss, (-1, 1)))
prob = gmm.predict_proba(tf.reshape(net1_loss, (-1, 1))) 
prob = prob[:,gmm.means_.argmin()]    
ind_labeled = prob > threshold
ind_unlabeled = prob < threshold

# co-divide
labeled_images, labeled_labels , unlabeled_images, unlabeled_labels= cifar10.co_divide(ind_labeled, ind_unlabeled) # not augmented
labeled_dataset = data2tensor(labeled_images, labeled_labels, BATCH_SIZE)
unlabeled_dataset = data2tensor(unlabeled_images, unlabeled_labels, BATCH_SIZE)

# augmentation
preds_labeled_list = []
preds_unlabeled_list = []

for i in range(M):
    labeled_dataset_augmented = augment(labeled_dataset, BATCH_SIZE)
    unlabeled_dataset_augmented = augment(unlabeled_dataset, BATCH_SIZE)
    
    preds_labeled = predict_batchwise(net1, labeled_dataset_augmented)
    preds_labeled_list.append(preds_labeled)

    preds_unlabeled_1 = predict_batchwise(net1, unlabeled_dataset_augmented)
    preds_unlabeled_list.append(preds_unlabeled_1)  

    preds_unlabeled_2 = predict_batchwise(net2, unlabeled_dataset_augmented)
    preds_unlabeled_list.append(preds_unlabeled_2)  
    
preds_labeled_list = np.array(preds_labeled_list)
preds_labeled_avg = np.mean(preds_labeled_list, axis = 0)
del preds_labeled_list

preds_unlabeled_list = np.array(preds_unlabeled_list)
preds_unlabeled_avg = np.mean(preds_unlabeled_list, axis = 0)
del preds_unlabeled_list

# co-refinement

# labels_refined = (W_P * (preds_labeled_list[0]) + (1 - W_P) * preds_labeled + W_P * (preds_labeled_list[1]) + (1 - W_P) * preds_labeled)/2  # make int
# labels_shapened =  sharpen(labels_refined, T)

# # co-guessing
# preds_unlabeled_1 = predict_batchwise(net1, unlabeled_dataset_augmented)
# preds_unlabeled_2 = predict_batchwise(net1, unlabeled_dataset_augmented)
# preds_refined =  0.5 * (preds_unlabeled_1 + preds_unlabeled_2)
# preds_sharpened = sharpen(preds_refined, T)


# 

