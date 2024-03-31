import tensorflow as tf
tf.random.set_seed(42)

'''
Since the target sequences are padded, it is important to apply a padding mask when
calculating the loss.
'''
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)



def accuracy_function(real, pred):
    # 'real' is token to number and is actually equivallent to the
    ## index value of the entire index array . ensure is tf.int32
    real = tf.cast(real, dtype=tf.int32)
    ### Extract the indices of maximum predictions and ensure it's also tf.int32
    predicted_indices = tf.cast(tf.argmax(pred, axis=2), dtype=tf.int32)
    ### True if value of real index and predicted index  is same
    accuracies = tf.equal(real, predicted_indices)
    ### generate mask from real where non zero value is True and Zero is False
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    #### True when both mask and accuracy  value is True
    accuracies = tf.math.logical_and(mask, accuracies)
    ### convert True in accuracy and mask  as 1 and rest all Zeo
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    ###
    total_accurate_prediction = tf.reduce_sum(accuracies)
    total_non_padding_tokens = tf.reduce_sum(mask)
    res =  total_accurate_prediction / total_non_padding_tokens
    return res

