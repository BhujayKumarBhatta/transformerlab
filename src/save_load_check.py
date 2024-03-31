
import tensorflow as tf
tf.random.set_seed(42)
import pickle
import os
import numpy as np
np.random.seed(123)

from attention_plots import plot_attention_weights, show_plot
from transformer_model import TransformerModelV2


model_path= '/mnt/d/MyDev/attention/models'

def save_load_check(model, model_class=TransformerModelV2, 
                model_path=model_path,
               num_layers=6,
               d_model=512,
               num_heads=8, dff=2048,
               en_vocab_size=1000,
               de_vocab_size=1000,
               en_seq_len=100,
               de_seq_len=150,
               idx=4,
               X1=None, X2=None, y=None,
               plot_function=plot_attention_weights,
               show_plot_function=show_plot,
               save=True, load=True, check=True, plot=True,
                    lang1_tokenizer=None, 
                    lang2_tokenizer=None,
                figsize=(10, 15), nrows=4, ncols=2
               ):    
    
    model_full_name = os.path.join(model_path, 'CustomTrans_L4_dm128_dff512_h8_acc87_DE500_Fit.weights.h5')
    if save:
        print('model before save:')
        print(model.summary())
        if check:
            tf.config.run_functions_eagerly(True)    
            output = model((X1[idx:idx+1], X2[idx:idx+1]), training=False)
            predicted_id = np.argmax(output, axis=-1)
            print('predicted_id:\n',  predicted_id)
            print('label:\n', y[idx:idx+1])
            print()
        model.save_weights(model_full_name)
        with open(model_path + '/config.pkl', 'wb') as config_file:
                pickle.dump(model.get_config(), config_file)
        with open(model_path + '/config.pkl', 'rb') as config_file:
                fit_model_config = pickle.load(config_file)
        
    if load:
        loaded_model = model_class(num_layers,
                                           d_model,
                                           num_heads, dff,
                                           en_vocab_size,
                                           de_vocab_size,
                                           en_seq_len,
                                           de_seq_len,                                   
                                           debug=False)
        loaded_model_out = loaded_model((X1[:2], X2[:2]), training=False)
        loaded_model.load_weights(model_full_name)
        if check:
            print('model loaded from saved weigts:')
            print(loaded_model.summary())
            output = loaded_model((X1[idx:idx+1], X2[idx:idx+1]), training=False)
            predicted_id = np.argmax(output, axis=-1)
            print('predicted_id:\n',  predicted_id)
            print('label:\n', y[idx:idx+1])
        if plot:
            if show_plot_function:
                show_plot(model=loaded_model, start=1, end=110, n=1,  ####sample_decoder_input[:,:-1]
                plt_function=plot_function, X1=X1, X2=X2, y=y, idx=idx,
                         lang1_tokenizer=lang1_tokenizer,
                         lang2_tokenizer=lang2_tokenizer,
                         figsize=figsize, nrows=nrows, ncols=ncols)
    # tf.config.run_functions_eagerly(False)
    return loaded_model

# loaded_model = save_load_check(trained_model, model_class=TransformerModelV2, 
#                 model_path=model_path,
#                num_layers=num_layers,
#                d_model=d_model,
#                num_heads=num_heads, dff=dff,
#                en_vocab_size=en_vocab_size,
#                de_vocab_size=de_vocab_size,
#                en_seq_len=en_seq_len,
#                de_seq_len=de_seq_len,
#                )