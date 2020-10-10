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
