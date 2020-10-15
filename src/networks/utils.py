import tensorflow as tf

@tf.function
def test_step(model, metrics, images, labels, all_metrics = False):
    '''
    return single batch loss
    '''
    cate_cross_entropy = metrics['cate_cross_entropy']
    acc = metrics['acc']
    test_loss = metrics['test_loss']

    predictions = model(images, training=False)
    
    
    loss_elementwise = cate_cross_entropy(labels, predictions)
    loss = test_loss(loss_elementwise)
    accuracy = acc(labels, predictions)
    
    if all_metrics:
        return loss_elementwise, loss, accuracy
    return loss_elementwise


def predict_batchwise(model, dataset):
    predicted = []
    for batch_images, batch_labels in dataset:
        predicted.append(model(batch_images, training = False))
        # print(predicted[-1])
    return tf.argmax(tf.concat(predicted, axis = 0), axis = 1).numpy()

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
            loss_all.append(test_step(model, metrics, batch_images, batch_labels, all_metrics))        
        

    loss_all = tf.concat(loss_all, -1)
    return loss_all


def data2tensor(train_images, train_labels, batch_size):
    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images/255, train_labels)).batch(batch_size=batch_size)

    return train_dataset

    

def normlize_loss(losses):
    return (losses-tf.math.reduce_min(losses)/(tf.math.reduce_max(losses)-tf.math.reduce_min(losses)))   


def augment(dataset, batch_size):
    images_combined = []
    labels_combined = []
    for images, labels in dataset:  
        iamges = tf.image.resize_with_crop_or_pad(images, 40, 40)
        images = tf.image.random_crop(iamges , [images.shape[0], 32, 32, 3])
        images = tf.image.per_image_standardization(images)
        images_combined.append(images)
        labels_combined.append(labels)
    images_combined = tf.concat(images_combined, axis = 0)
    print(images_combined.shape)
    labels_combined = tf.concat(labels_combined, axis = 0)
    print(labels_combined.shape)
    dataset = tf.data.Dataset.from_tensor_slices((images_combined, labels_combined)).batch(batch_size=batch_size)
    return dataset


def sharpen(preds, temp):
    preds = preds**(1/temp)
    preds = preds / preds.sum() # normalize
    return preds           