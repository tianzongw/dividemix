from data.data_loader import dataset
from networks.network import get_model, train_model
from networks.utils import samplewise_loss, data2tensor


cifar10 = dataset("cifar10")

cifar10.noisify("symm", ratio= 0.4)


net1 =  get_model("preact", 32, 32, 3)
net2 =  get_model("preact", 32, 32, 3)


#warmup

train_model(net1, cifar10.train_images, cifar10.train_labels, 8, 10)
cifar10_dataset = data2tensor(cifar10.train_images, cifar10.train_labels, 8)
net1_loss = samplewise_loss(net1, cifar10_dataset, all_metrics=False)
print(net1_loss)