from data.data_loader import dataset
from networks.network import get_model, train_model

cifar10 = dataset("cifar10")

cifar10.noisify("symm", ratio= 0.4)


net1 =  get_model("preact", 32, 32, 3)
net2 =  get_model("preact", 32, 32, 3)


#warmup

train_model(net1, cifar10.train_images, cifar10.train_labels, 8, 10)