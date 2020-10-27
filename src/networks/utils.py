import tensorflow as tf
import numpy as np

@tf.function
def test_step(model, metrics, images, labels, all_metrics = False):
    '''
    return single batch loss
    '''
    cate_cross_entropy = metrics['cate_cross_entropy']
    acc = metrics['acc']
    test_loss = metrics['test_loss']

    predictions = model(images, training=True)
    
    
    loss_elementwise = cate_cross_entropy(labels, predictions)
    loss = test_loss(loss_elementwise)
    accuracy = acc(labels, predictions)
    
    if all_metrics:
        return loss_elementwise, loss, accuracy
    return loss_elementwise


def predict_batchwise(model, dataset):
    predicted = []
    try:
        for batch_images, _ in dataset:
            predicted.append(model(batch_images, training = True).numpy())
    except Exception as e:
        # print(e)
        for batch_images in dataset:
            predicted.append(model(batch_images, training = True).numpy())
    # print(len(predicted))
    return np.concatenate(predicted, axis = 0)


def samplewise_loss(model, dataset, all_metrics = True):
    '''
    return all losses
    '''
    metrics = {'cate_cross_entropy' : tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE),
               'acc' : tf.keras.metrics.SparseCategoricalAccuracy(),
               'test_loss': tf.keras.metrics.Mean()
              }

    loss_all = []

    i = 0
    for batch_images, batch_labels in dataset:
        i+=1
        if all_metrics:
            loss_all.append(test_step(model, metrics, batch_images, batch_labels, all_metrics)[0])
        else:
            res = test_step(model, metrics, batch_images, batch_labels, all_metrics)
            # print("res")
            # print(res)
            n_res = normlize_loss(res)
            # print("nres")
            # print(n_res)
            # print("____")
            loss_all.append( n_res )    
        

    loss_all = tf.concat(loss_all, -1)
    return loss_all


def data2tensor(train_images, train_labels, batch_size):
    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size=batch_size)

    return train_dataset

    

def normlize_loss(losses):
    return ((losses-tf.math.reduce_min(losses))/(tf.math.reduce_max(losses)-tf.math.reduce_min(losses)))   

import matplotlib.pyplot as plt

def augment(dataset, batch_size, mode = "train"):
    images_combined = []
    labels_combined = []
    for images, labels in dataset: 
        if mode == 'train':
            images_new = []
            for image in images:
                image = tf.image.resize_with_crop_or_pad(image, 36, 36)
                
                image =  tf.image.random_crop(image , [32, 32, 3])
                images_new.append(image)
            # img_np = images[5].numpy()
            # print(np.min(img_np), np.max(img_np))

            # plt.imsave( "a.png", img_np/255.)
            # print(images[-1].shape)
            # print(len(images))
            images = tf.stack(images_new)
            images = tf.image.random_flip_left_right(images, seed=None)
        images = tf.image.per_image_standardization(images)
        images_combined.append(images)
        labels_combined.append(labels)
    images_combined = tf.concat(images_combined, axis = 0)
    labels_combined = tf.concat(labels_combined, axis = 0)
    dataset = tf.data.Dataset.from_tensor_slices((images_combined, labels_combined)).batch(batch_size=batch_size)
    return dataset


def sharpen(preds, temp):
    preds = preds**(1/temp)
    preds = preds / preds.sum() # normalize
    return preds           


def extract_img_from_dataset(dateset):
    images = []
    for img, _ in dateset:
        images.append(img)
    
    return np.concatenate(images)


def linear_rampup(current, warm_up, rampup_length=16, lambda_u = 25):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u*float(current)

def compare_to_mask(mask1, mask2):
    return sum(mask1*2-1 == mask2 ), sum(mask1), sum(mask2)
    