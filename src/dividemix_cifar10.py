import tensorflow as tf
import numpy as np
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from data.data_loader import dataset
from networks.network import get_model, warm_up, predict_model
from networks.utils import augment, compare_to_mask, samplewise_loss, data2tensor, normlize_loss, predict_batchwise, sharpen, extract_img_from_dataset, linear_rampup, manual_acc
from sklearn.mixture import GaussianMixture

threshold  = 0.5
BATCH_SIZE = 512
WARMUP_EPOCH = 10
MAX_EPOCH = 100
W_P = 0.5
T = 0.1
M = 2
alpha = 4

cifar10 = dataset("cifar10")

cifar10.noisify("symm", ratio= 0.1)
cifar10_dataset = data2tensor(cifar10.train_images, cifar10.train_labels, BATCH_SIZE)
test_dataset = data2tensor(cifar10.test_images, cifar10.test_labels, BATCH_SIZE)
cifar10_augmented = augment(cifar10_dataset, BATCH_SIZE)

net1 =  get_model("preact", 32, 32, 3)
net2 =  get_model("preact", 32, 32, 3)

test_dataset_normalized = augment(test_dataset, BATCH_SIZE, mode = 'test')


# warmup
try: 
    # net1.load_weights('./models/checkpoint1')
    # net2.load_weights('./models/checkpoint2')
    print("train 1")
    warm_up(net1, cifar10_dataset,  BATCH_SIZE, WARMUP_EPOCH, test_dataset_normalized)
    print("train 2")
    warm_up(net2, cifar10_dataset,  BATCH_SIZE, WARMUP_EPOCH, test_dataset_normalized)
    net1.save_weights('./models/checkpoint1')
    net2.save_weights('./models/checkpoint2')

except:
    warm_up(net1, cifar10_augmented,  BATCH_SIZE, WARMUP_EPOCH)
    warm_up(net2, cifar10_augmented,  BATCH_SIZE, WARMUP_EPOCH)
    net1.save_weights('./models/checkpoint1')
    net2.save_weights('./models/checkpoint2')




def divmix_step(net1, net2, epoch):
    # model per-sample loss 
    net2_loss = samplewise_loss(net2, cifar10_augmented, all_metrics=False)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(tf.reshape(net2_loss, (-1, 1)))
    prob = gmm.predict_proba(tf.reshape(net2_loss, (-1, 1))) 
    prob = prob[:,gmm.means_.argmin()]    
    ind_labeled = prob > threshold
    ind_unlabeled = prob < threshold
    w_x = prob[ind_labeled].reshape(-1, 1)
    print("Data split:")
    print(sum(ind_labeled), sum(ind_unlabeled))
    print("noise data identified: ", compare_to_mask(ind_unlabeled, cifar10.mask_used))
    
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
    labels_refined = (1 - w_x) * (preds_labeled_avg) +  w_x * labeled_labels_onehot
    labels_shapened = sharpen(labels_refined, T)

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


    mixed_input_x, mixed_input_u = mixed_input[ : len(labels_shapened)*2, :], mixed_input[len(labels_shapened)*2 : , :]
    mixed_target_x, mixed_target_u = mixed_target[ : len(labels_shapened)*2, :], mixed_target[len(labels_shapened)*2 : , :]

    mixed_input_x = tf.convert_to_tensor(mixed_input_x, dtype=tf.float32)
    mixed_input_u = tf.convert_to_tensor(mixed_input_u, dtype=tf.float32)
    mixed_target_x = tf.convert_to_tensor(mixed_target_x, dtype=tf.float32)
    mixed_target_u = tf.convert_to_tensor(mixed_target_u, dtype=tf.float32)

    mixed_dataset_x = tf.data.Dataset.from_tensor_slices((mixed_input_x, mixed_target_x)).batch(batch_size=BATCH_SIZE)
    mixed_dataset_u = tf.data.Dataset.from_tensor_slices((mixed_input_u, mixed_target_u)).batch(batch_size=BATCH_SIZE)


    Lx = tf.keras.losses.CategoricalCrossentropy()
    Lu = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(lr = 0.02)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    
    step = 0
    iter_mixed_dataset_u = iter(mixed_dataset_u)

    for mixed_input_x_batch, mixed_target_x_batch in mixed_dataset_x:
    
        try:
            mixed_input_u_batch, mixed_target_u_batch = iter_mixed_dataset_u.get_next()
        except:
            iter_mixed_dataset_u = iter(mixed_dataset_u)
            mixed_input_u_batch, mixed_target_u_batch = iter_mixed_dataset_u.get_next()

        # regularization
        # prior = torch.ones(args.num_class)/args.num_class
        # pred_mean = torch.softmax(logits, dim=1).mean(0)
        # penalty = torch.sum(prior*torch.log(prior/pred_mean))

        with tf.GradientTape() as tape:
            logits_x_batch = net1(mixed_input_x_batch, training=True)
            logits_u_batch = net1(mixed_input_u_batch, training=True)

            num_class = logits_x_batch.shape[1]
            # stack above 2 logits
            logits_batch = tf.concat([logits_x_batch, logits_u_batch], 0)

            # take mean by 0 axis as pred_mean
            pred_mean = tf.math.reduce_mean(tf.nn.softmax(logits_batch, axis=1), axis = 0)

            # create a uniform prior regarding to # classes aas prior
            prior = tf.ones(num_class)/num_class
            
            # compute penalty = torch.sum(prior*torch.log(prior/pred_mean))
            penalty = tf.math.reduce_sum(prior*tf.math.log(prior/pred_mean))

            loss = Lx(y_true=mixed_target_x_batch, y_pred=logits_x_batch) \
                 + Lu(y_true=mixed_target_u_batch, y_pred=logits_u_batch) * linear_rampup(epoch, WARMUP_EPOCH) \
                 + penalty
            
        gradients = tape.gradient(loss, net1.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, net1.trainable_variables))

        train_accuracy(mixed_target_x_batch, logits_x_batch)
        step += 1

        print('epoch', epoch+1, 'batch', step, 'acc:', train_accuracy.result())

acc_list_1 = []
acc_list_2 = []

print('before training')
predict_model(net1,test_dataset)
predict_model(net2,test_dataset)

for epoch in range(MAX_EPOCH):
    
    print('epoch', epoch+1)
    divmix_step(net1, net2, epoch)
    divmix_step(net2, net1, epoch)
    
    acc_list_1.append(predict_model(net1, test_dataset))
    acc_list_2.append(predict_model(net2, test_dataset))
    print(acc_list_1)
    print(acc_list_2)



