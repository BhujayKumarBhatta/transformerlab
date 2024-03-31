import pathlib
import os
import re
import string
import time
import random
random.seed(42)
from datetime import datetime
import tensorflow as tf
tf.random.set_seed(42)
from transformer_model import TransformerModel
from lr_schedular import  CustomSchedule, MyCustomSchedule
from masked_loss_accu import loss_function, accuracy_function
en_vocab_size = 0
en_seq_len = 0
de_vocab_size = 0
de_seq_len= 0
USE_PE = False
train_data = None
COLAB = False


# @tf.function(reduce_retracing=True)  ### can not do incremental training with this 
def train_step(inp, tar, label, transformer, optimizer, loss_function, accuracy_function):
    # tar_inp = tar[:, :-1]
    # tar_real = tar[:, 1:]  #### without the 'START OR SOS' tag
    with tf.GradientTape() as tape:
        predictions= transformer([inp, tar], training=True)
        ## #### predictions should be without the 'START OR SOS' tag
        loss = loss_function(label, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    accuracy = accuracy_function(label, predictions)
    return loss, accuracy


def start_training(X1, X2, y, transformer, n_sentences=500,
    num_layers=6, d_model=512, num_heads=8, dff=2048,
    # en_vocab_size=en_vocab_size,
    # de_vocab_size=de_vocab_size,
    # en_seq_len=en_seq_len, de_seq_len=de_seq_len,
    EPOCHS=10, BATCH_SIZE=64, TRAIN_BATCHES=8,
    pos_dropout=0.1, en_dropout=0.1,
    de_dropout=0.1, patience=5,
    model_path= '/content/drive/My Drive/writeups/notebooks/models',
    custom_lrsched=True,
    lr=0.001,
    disp_metric_on=10,
    tf_data=USE_PE,
    data=train_data,
    COLAB=COLAB,
    custom_scheduler=MyCustomSchedule,
    shuffle_BUFFER_SIZE=None,
    save_model=True,
    ):

    if not COLAB:
      model_path= '/mnt/d/MyDev/attention/models'
    # Adjust based on your dataset size and GPU memory
    if shuffle_BUFFER_SIZE is None:
        BUFFER_SIZE = len(X1)  # Assuming X1 and X2 are your dataset
    else:
        BUFFER_SIZE = shuffle_BUFFER_SIZE
    
    ## Assuming X1 and X2 are numpy arrays or similar structures
    ## Convert them into TensorFlow datasets and batch them appropriately
    if tf_data:
        dataset = data
        print('tf dataset  used ')
    else:
        dataset = tf.data.Dataset.from_tensor_slices(((X1, X2), y))
        dataset = dataset.shuffle(
            BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)
    total_batches = (n_sentences) // BATCH_SIZE
    # total_batches = tf.data.experimental.cardinality(dataset).numpy()
    print('total_batches:', total_batches)
    if not TRAIN_BATCHES:
        TRAIN_BATCHES = total_batches
    train_data = dataset.take(TRAIN_BATCHES).cache()
    # train_data = dataset
    train_batches = tf.data.experimental.cardinality(train_data).numpy()
    print('train_batches:', TRAIN_BATCHES )
    if custom_lrsched:
        learning_rate = custom_scheduler(d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    else:
        learning_rate = lr
        optimizer = tf.keras.optimizers.Adam(learning_rate)   
    best_accuracy = 0.0  # Keep track of best accuracy
    wait = 0  # Counter for patience
    stop_training = False  # Flag to stop training
    current_date = datetime.now().strftime('%Y-%m-%d')

    for epoch in range(EPOCHS):
        if stop_training:
            break
        start = time.time()
        total_loss = 0
        total_accuracy = 0
        dot_cnt = 1        
        for (batch, ((inp, tar_inp), tar_real)) in enumerate(train_data):           
            loss, accuracy = train_step(inp, tar_inp, tar_real, 
                                        transformer, optimizer,
                                        loss_function,
                                        accuracy_function)
            total_accuracy += accuracy
            total_loss += loss
            if batch % disp_metric_on == 0:
                print(f'\r{"." * dot_cnt}', end='')
                print(f',loss: {loss:.4f} - accuracy: {accuracy:.4f}', end=' ')
                dot_cnt += 1
         # Calculate average loss and accuracy over the epoch
        total_loss /= train_batches
        total_accuracy /= train_batches
        # Checkpoint logic
        if total_accuracy > best_accuracy:
            best_accuracy = total_accuracy
            model_path= model_path
            model_name = f'transformer_{datetime.now().strftime("%Y-%m-%d")}_{num_layers}_{d_model}_{num_heads}_{dff}_{epoch:02d}_{total_accuracy:.2f}.weights.h5'
            model_name = os.path.join(model_path, model_name)
            if save_model:
                transformer.save_weights(model_name)
                print(f'saved modeld: {model_name}')
            wait = 0  # Reset wait counter            
        else:
            wait += 1
        # Early stopping logic
        if wait >= patience:
            print("Early stopping due to no improvement in loss.")
            stop_training = True
        # print(f'\rEpoch {epoch + 1}, Loss: {total_loss.numpy():.4f}, Accuracy: {total_accuracy.numpy():.4f}', end='    \n')
        print(f'\rEpoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {total_accuracy:.4f}', end='    \n')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    return transformer