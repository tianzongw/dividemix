import tensorflow as tf
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models

class dataset():
    
    def __init__(self, name, num_samples = None):
        
        self.asym_map = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8}
        
        if name == 'cifar10':
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
        elif name == 'cifar100':
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar100.load_data()

        if num_samples is not None:
            self.train_images = self.train_images[:num_samples]
            self.train_labels = self.train_labels[:num_samples]

        self.train_images = self.train_images/255.
        self.test_images = self.test_images/255.
        
    def noisify(self, mode, ratio = 0.4):
        
        mask = np.random.choice(2, len(self.train_labels), p = [1 - ratio, ratio])
        self.mask_used = mask
        self.noisy_train_labels = self.train_labels.copy()
        
        for i in range(len(self.train_labels)):          
            if mask[i]:
                if mode == 'symm':
                    self.noisy_train_labels[i] = random.randint(0, len(np.unique(self.train_labels)))
                elif mode == 'asymm':
                    self.noisy_train_labels[i] = self.asym_map[self.train_labels[i][0]]
    
    def co_divide(self, labeled_idx, unlabeled_idx):

        labeled_images = self.train_images[labeled_idx]
        labeled_labels = self.train_labels[labeled_idx].reshape(-1)
        labeled_labels_onehot = to_categorical(labeled_labels)
        
        unlabeled_images = self.train_images[unlabeled_idx]
        unlabeled_labels = self.train_labels[unlabeled_idx].reshape(-1)
        unlabeled_labels_onehot = to_categorical(unlabeled_labels)


        return labeled_images, labeled_labels, unlabeled_images, unlabeled_labels, labeled_labels_onehot, unlabeled_labels_onehot