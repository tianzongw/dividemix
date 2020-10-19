import tensorflow as tf
import numpy as np

from data.data_loader import dataset
from networks.network import get_model, train_model
from networks.utils import augment, samplewise_loss, data2tensor, normlize_loss, predict_batchwise, sharpen, extract_img_from_dataset
from sklearn.mixture import GaussianMixture

threshold  = 0.4
BATCH_SIZE = 1024
WARMUP_EPOCH = 10
MAX_EPOCH = 2
W_P = 0.5
T = 0.1
M = 2
alpha = 4

cifar10 = dataset("cifar10")


cifar10.noisify("symm", ratio= 0.1)
cifar10_dataset = data2tensor(cifar10.train_images, cifar10.train_labels, BATCH_SIZE)


cifar10_augmented = augment(cifar10_dataset, BATCH_SIZE)
net1 =  get_model("preact", 32, 32, 3)

net2 = net1
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
labeled_images, labeled_labels, unlabeled_images, unlabeled_labels, labeled_labels_onehot, unlabeled_labels_onehot= cifar10.co_divide(ind_labeled, ind_unlabeled) # not augmented
labeled_dataset = data2tensor(labeled_images, labeled_labels, BATCH_SIZE)
unlabeled_dataset = data2tensor(unlabeled_images, unlabeled_labels, BATCH_SIZE)

# augmentation
preds_labeled_list = []
preds_unlabeled_list = []
all_inputs = []

for i in range(M):
    labeled_dataset_augmented = augment(labeled_dataset, BATCH_SIZE)
    all_inputs.append(extract_img_from_dataset(labeled_dataset_augmented))

    preds_labeled = predict_batchwise(net1, labeled_dataset_augmented)
    preds_labeled_list.append(preds_labeled)


for i in range(M):
    unlabeled_dataset_augmented = augment(unlabeled_dataset, BATCH_SIZE) 
    all_inputs.append(extract_img_from_dataset(unlabeled_dataset_augmented))


    preds_unlabeled_1 = predict_batchwise(net1, unlabeled_dataset_augmented)
    preds_unlabeled_list.append(preds_unlabeled_1)  

    preds_unlabeled_2 = predict_batchwise(net2, unlabeled_dataset_augmented)
    preds_unlabeled_list.append(preds_unlabeled_2)


preds_labeled_list = np.array(preds_labeled_list)
preds_labeled_avg = np.mean(preds_labeled_list, axis = 0)
del(preds_labeled_list)

preds_unlabeled_list = np.array(preds_unlabeled_list)
preds_unlabeled_avg = np.mean(preds_unlabeled_list, axis = 0)
del(preds_unlabeled_list)

# co-refinement
labels_refined = (W_P * (preds_labeled_avg) + (1 - W_P) * labeled_labels_onehot)
labels_shapened =  sharpen(labels_refined, T)

# co-guessing
preds_sharpened = sharpen(preds_unlabeled_avg, T)


# mixmatch
all_inputs = np.concatenate(all_inputs)
all_targets = np.concatenate([labels_shapened, labels_shapened, preds_sharpened, preds_sharpened])

idx = np.random.shuffle(np.arange(all_inputs.shape[0]))

input_a, input_b = all_inputs, np.squeeze(all_inputs[idx])
target_a, target_b = all_targets, np.squeeze(all_targets[idx])

l = np.random.beta(alpha, alpha)        
l = max(l, 1-l)

mixed_input = l * input_a + (1 - l) * input_b  
mixed_target = l * target_a + (1 - l) * target_b

subset_number = 40
subset_size = int(len(mixed_input) / subset_number)

logits = []
for i in range(subset_number):
    subset_images = mixed_input[i * subset_size: (i+1)*subset_size ]
    subset_images = tf.convert_to_tensor(subset_images, dtype=tf.float32)
    subset_dataset = tf.data.Dataset.from_tensor_slices((subset_images)).batch(batch_size=BATCH_SIZE)
    logits.append(predict_batchwise(net1, subset_dataset))
    
logits = np.concatenate(logits)

logits_x, logits_u = logits[ : len(labels_shapened)*2, :], logits[len(labels_shapened)*2 : , :]
mixed_target_x, mixed_target_u = mixed_target[ : len(labels_shapened)*2, :], mixed_target[len(labels_shapened)*2 : , :]


logits_x = tf.convert_to_tensor(logits_x, dtype=tf.float32)
print(logits_x.shape)
logits_x_dataset = tf.data.Dataset.from_tensor_slices((logits_x)).batch(batch_size=BATCH_SIZE)

mixed_target_x = tf.convert_to_tensor(mixed_target_x, dtype=tf.float32)
print(mixed_target_x.shape)

mixed_target_x_dataset = tf.data.Dataset.from_tensor_slices((mixed_target_x)).batch(batch_size=BATCH_SIZE)



mixed_target_x_dataset = iter(mixed_target_x_dataset)
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adadelta()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

for logits_x_batch in logits_x_dataset:  
    mixed_target_x_batch = mixed_target_x_dataset.get_next()
    print(mixed_target_x_batch.shape)
    with tf.GradientTape() as tape:
        loss = loss_object(y_true=mixed_target_x_batch, y_pred=logits_x_batch)
    gradients = tape.gradient(loss, net1.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, net1.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    print("update worked")
    break






