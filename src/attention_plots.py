import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random
random.seed(42)
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(42)

lang1_tokenizer, lang2_tokenizer = None,  None






def id_to_token_in_vocab(ids, tokenizer=lang1_tokenizer):
    # Convert a single ID to a sequence by wrapping it in a list of lists
    sequences = [[id] for id in ids]
    # Use sequences_to_texts to convert these sequences back to text
    texts = tokenizer.sequences_to_texts(sequences)
    # sequences_to_texts returns a list of text strings
    return texts

def id_to_token_out_vocab(ids, tokenizer=lang2_tokenizer):
    sequences = [[id] for id in ids]
    texts = tokenizer.sequences_to_texts(sequences)
    return texts


def plot_attention_weights(encoder_input, decoder_input,
                           attention_weights, layer=None,
                           lang1_tokenizer=lang1_tokenizer,
                           lang2_tokenizer=lang2_tokenizer,
                          figsize=(12, 15),
                          nrows=2, ncols=4):
    # Assuming 'attention_weights[layer]' is of shape (num_heads, seq_len_decoder, seq_len_encoder)
    num_heads = attention_weights.shape[0]   #attention_weights[layer].shape[0]
    seq_len_decoder = attention_weights.shape[1]  ### attention_weights[layer].shape[1]
    seq_len_encoder = attention_weights.shape[2]

    # Set up the matplotlib figure and axes, based on the number of heads
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize)

    # If there's only one head, matplotlib doesn't return an array of axes
    if num_heads == 1:
        axes = np.array([axes])

    encoder_input_tokens = id_to_token_in_vocab(encoder_input[0], lang1_tokenizer)
    decoder_input_tokens = id_to_token_out_vocab(decoder_input[0], lang2_tokenizer)

    # Flatten the list of tokens if they are returned as list of lists
    encoder_input_tokens = [token for sublist in encoder_input_tokens for token in sublist.split()]
    decoder_input_tokens = [token for sublist in decoder_input_tokens for token in sublist.split()]
    len_en_token = len(encoder_input_tokens)
    len_de_token = len(decoder_input_tokens)
    for head, ax in zip(range(num_heads), axes.flatten()):
        # ax = axes[head]
        # attention_head_weights = attention_weights[layer][head]
        # print('head:', head)
        attention_head_weights = attention_weights[head]
        # print('attention_head_weights:', attention_head_weights.shape)
        cax = ax.matshow(attention_head_weights[:len_de_token, :len_en_token], cmap='viridis')

        ax.set_xticks(range(len(encoder_input_tokens)))
        ax.set_yticks(range(len(decoder_input_tokens[1:])))

        ax.set_xticklabels(encoder_input_tokens, rotation=90)
        ax.set_yticklabels(decoder_input_tokens[1:])

        ax.set_title(f'Head {head+1}')

    # plt.colorbar(cax, ax=axes, orientation='vertical', fraction=0.025, pad=0.04)
    plt.tight_layout()
    plt.show()



def show_plot(model, start=1, end=110, n=1,  ####sample_decoder_input[:,:-1]
             layer='decoder_layer5_block2', plt_function=plot_attention_weights,
             X1=None, X2=None, y=None, idx=3, lang1_tokenizer=None, lang2_tokenizer=None,
             figsize=(15, 15), nrows=2, ncols=4):
    random_integers = [random.randint(start, end) for _ in range(n)]
    for n in random_integers:
        print(n)
        sample_encoder_input = X1[n:n+1]  # Select a sample input
        sample_decoder_input = X2[n:n+1] ###[:,:-1]  # Select a corresponding target input
        print('sample_decoder_input:\n', sample_decoder_input)
        tf.config.run_functions_eagerly(True)
        prediction_out = model(
            [sample_encoder_input, sample_decoder_input], training=False)
        print('prediction_out:\n', prediction_out.shape)
        predicted_id = np.argmax(prediction_out, axis=-1)
        print('predicted_id:\n', predicted_id)
        print('predicted label:\n', y[n:n+1])
        attention_weights = model.attention_weights
        layer = list(attention_weights.keys())[-1]
        attention_weights_n = attention_weights[layer].numpy()
        # print('attention_weights_n:', attention_weights_n)
        tf.config.run_functions_eagerly(False)
        # attention_weights_n = trained_transformer.decoder.dec_layers[-1].last_attn_scores
        # attention_weights_n = tf.squeeze(attention_weights_n, axis=0)
        print('attn_scores: ' , attention_weights_n.shape)
        plt_function(sample_encoder_input, 
                               sample_decoder_input, 
                               attention_weights_n,
                                lang1_tokenizer=lang1_tokenizer,
                    lang2_tokenizer=lang2_tokenizer,
                    figsize=figsize, nrows=nrows, ncols=ncols)