import tensorflow as tf
import numpy as np
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from data.data_loader import dataset
from networks.network import get_model, train_model, predict_model
from networks.utils import augment, samplewise_loss, data2tensor, normlize_loss, predict_batchwise, sharpen, extract_img_from_dataset, linear_rampup
from sklearn.mixture import GaussianMixture

threshold  = 0.4
BATCH_SIZE = 512
WARMUP_EPOCH = 30
MAX_EPOCH = 100
W_P = 0.5
T = 0.1
M = 2
alpha = 4

cifar10 = dataset("cifar10")


cifar10.noisify("symm", ratio= 0.1)
cifar10_dataset = data2tensor(cifar10.train_images, cifar10.train_labels, BATCH_SIZE)


cifar10_augmented = augment(cifar10_dataset, BATCH_SIZE)
net1 =  get_model("preact", 32, 32, 3)
#net2 = net1
net2 =  get_model("preact", 32, 32, 3)

test_dataset = data2tensor(cifar10.test_images, cifar10.test_labels, BATCH_SIZE)

# warmup
try: 
    # net1.load_weights('./models/checkpoint')
    train_model(net1, cifar10_augmented,  BATCH_SIZE, WARMUP_EPOCH)
    train_model(net2, cifar10_augmented,  BATCH_SIZE, WARMUP_EPOCH)
    net1.save_weights('./models/checkpoint1')
    net2.save_weights('./models/checkpoint2')

except:
    train_model(net1, cifar10_augmented,  BATCH_SIZE, WARMUP_EPOCH)
    net1.save_weights('./models/checkpoint')

# for epoch in MAX_EPOCH:

# print('load weights')

predict_model(net1,test_dataset)
predict_model(net2,test_dataset)

def divmix_step(net1, net2, epoch):
    # model per-sample loss 
    net1_loss = samplewise_loss(net1, cifar10_augmented, all_metrics=False)
    net1_loss = normlize_loss(net1_loss)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(tf.reshape(net1_loss, (-1, 1)))
    prob = gmm.predict_proba(tf.reshape(net1_loss, (-1, 1))) 
    prob = prob[:,gmm.means_.argmin()]    
    ind_labeled = prob > threshold
    ind_unlabeled = prob < threshold
    print(sum(ind_labeled), sum(ind_unlabeled))
    # co-divide
    labeled_images, labeled_labels, unlabeled_images, unlabeled_labels, labeled_labels_onehot, unlabeled_labels_onehot= cifar10.co_divide(ind_labeled, ind_unlabeled) # not augmented
    labeled_dataset = data2tensor(labeled_images, labeled_labels, BATCH_SIZE)
    unlabeled_dataset = data2tensor(unlabeled_images, unlabeled_labels, BATCH_SIZE)

    # print('before augment')

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

    # print('after augment')

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


    mixed_input_x, mixed_input_u = mixed_input[ : len(labels_shapened)*2, :], mixed_input[len(labels_shapened)*2 : , :]
    mixed_target_x, mixed_target_u = mixed_target[ : len(labels_shapened)*2, :], mixed_target[len(labels_shapened)*2 : , :]



    mixed_input_x = tf.convert_to_tensor(mixed_input_x, dtype=tf.float32)
    # mixed_input_x_dataset = tf.data.Dataset.from_tensor_slices((mixed_input_x)).batch(batch_size=BATCH_SIZE)

    mixed_input_u = tf.convert_to_tensor(mixed_input_u, dtype=tf.float32)


    mixed_target_x = tf.convert_to_tensor(mixed_target_x, dtype=tf.float32)

    mixed_target_u = tf.convert_to_tensor(mixed_target_u, dtype=tf.float32)



    mixed_dataset_x = tf.data.Dataset.from_tensor_slices((mixed_input_x, mixed_target_x)).batch(batch_size=BATCH_SIZE)
    mixed_dataset_u = tf.data.Dataset.from_tensor_slices((mixed_input_u, mixed_target_u)).batch(batch_size=BATCH_SIZE)

    iter_mixed_dataset_u = iter(mixed_dataset_u)
    # iter_mixed_dataset_x = iter(mixed_dataset_x)



    Lx = tf.keras.losses.CategoricalCrossentropy()
    Lu = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adadelta()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    i = 0

    for mixed_input_x_batch, mixed_target_x_batch in mixed_dataset_x:
    # for mixed_input_u_batch, mixed_target_u_batch in mixed_dataset_u:
    
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
            # print(logits_batch.shape)
            # print('stack_worked')

            # take mean by 0 axis as pred_mean
            pred_mean = tf.math.reduce_mean(tf.nn.softmax(logits_batch, axis=1), axis = 0)
            # print(pred_mean.shape)
            # print('softmax+mean worked')

            # create a uniform prior regarding to # classes aas prior
            prior = tf.ones(num_class)/num_class
            # print(prior)
            # print('prior worked')
            
            # compute penalty = torch.sum(prior*torch.log(prior/pred_mean))
            penalty = tf.math.reduce_sum(prior*tf.math.log(prior/pred_mean))
            # print(penalty)
            # print('penalty worked')

            loss = Lx(y_true=mixed_target_x_batch, y_pred=logits_x_batch) +\
                Lu(y_true=mixed_target_u_batch, y_pred=logits_u_batch) * linear_rampup(epoch, WARMUP_EPOCH) +\
                penalty
            
        gradients = tape.gradient(loss, net1.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, net1.trainable_variables))

        # train_loss(loss)
        # print('loss:', train_loss.result())
        train_accuracy(mixed_target_x_batch, logits_x_batch)
        i += 1

        print('epoch', epoch, 'batch', i, 'acc:', train_accuracy.result())

acc_list_1 = []
acc_list_2 = []
for epoch in range(MAX_EPOCH):
    divmix_step(net1, net2, epoch)
    divmix_step(net2, net1, epoch)
    
    acc_list_1.append(predict_model(net1, test_dataset))
    acc_list_2.append(predict_model(net2, test_dataset))
    print(acc_list_1)
    print(acc_list_2)



