import pathlib
import os
import re
import string
from unicodedata import normalize
import pandas as pd
from datetime import datetime

import time
import random
random.seed(42)
from random import randint
from pathlib import Path
import pathlib
import scipy
import numpy as np
np.random.seed(123)

import tensorflow as tf
tf.random.set_seed(42)


def encode_pos_sin_cosine(positions, embedding_dim, debug=False):
    exponent_i = np.arange(embedding_dim)
    if debug: print('exponent_i: ', exponent_i.shape)
    denominator = np.float32(embedding_dim)
    if debug: print('denominator scalar: ', denominator)
    angle_rates = 1 / np.power(10000, (2 * (exponent_i // 2)) / denominator )
    if debug: print('angle_rates:', angle_rates.shape)
    if isinstance(positions, int):
      positions = np.arange(positions)
    angle_rads = positions[:, np.newaxis] * angle_rates
    if debug: print('angle_rads:', angle_rads.shape)
    # Apply sin to even indices and cos to odd indices in the array
    sine_values_even = np.sin(angle_rads[:, 0::2])
    if debug: print('sine_values_even:',  sine_values_even.shape)
    cosine_values_odd = np.cos(angle_rads[:, 1::2])
    if debug: print('cosine_values_odd:', cosine_values_odd.shape)
    ### reorganizing the even and odd values
    sin_plus_cos_interleaved = np.empty(angle_rads.shape)
    sin_plus_cos_interleaved[:, 0::2] = sine_values_even
    sin_plus_cos_interleaved[:, 1::2] = sine_values_even
    # sin_plus_cosine_values = np.concatenate([sine_values_even, cosine_values_odd], axis=1)
    if debug: print('sin_plus_cos_interleaved:', sin_plus_cos_interleaved.shape)
    return sin_plus_cos_interleaved




# Define a custom layer for Positional Embedding
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model, max_pos=2048,
              pos_dropout=0.1, **kwargs):
    super().__init__(**kwargs)  # Initialize the superclass (Layer)
    self.d_model = d_model  # Store the dimensionality of the model embeddings
    '''
    # Create an embedding layer for tokens/words with masking for zero padding
    After passing the input x through the self.embedding layer, each token index is replaced by a d_model-dimensional embedding vector,
    resulting in a 3D tensor of shape [batch_size, sequence_length, d_model]
    '''
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
    '''
    # Generate the positional encoding matrix with the specified length and depth
     precomputes a matrix of positional encodings up to a length of 2048, which is presumed to be longer than any input sequence it will encounter.
    '''
    self.pos_encoding = encode_pos_sin_cosine(max_pos, d_model, debug=False)
    '''
    This tensor is scaled by the square root of d_model to normalize the variance
    '''
  # Define a method to compute the mask for padded tokens (to ignore them in computations)
    self.dropout = tf.keras.layers.Dropout(pos_dropout)

  def compute_mask(self, *args, **kwargs):
    '''
    In the context of the PositionalEmbedding class, the compute_mask method is designed to generate a mask tensor that indicates which items in the input are padding and should not be used in computations, such as attention weights. This is typically necessary when dealing with variable-length sequences padded to a common length for batch processing.

    The compute_mask method is calling the compute_mask method of the Embedding layer, which automatically generates a mask based on which inputs are equal to zero (since mask_zero=True was set when creating the Embedding layer).

    Here's an example to illustrate this:
      Sentence 1 (actual length 3): [23, 45, 67, 0, 0]
      Sentence 2 (actual length 4): [89, 34, 21, 66, 0]
    The mask is a boolean tensor with the same shape as the input sequences (excluding the embedding dimension). Each True value indicates a real token that should be processed, while each False indicates a padding token that should be ignored.
      Mask for Sentence 1: [True, True, True, False, False]
      Mask for Sentence 2: [True, True, True, True, False]

      The padded embeddings  even if they consists of zeros , when they goes through the linera layers to get the Q,K and V , the weights for the Q, K V may convert eh zero to noz zero . Therefore the dot products of q, k, v may result into some valid scores. Therefore the boolean mask will help to forcefully set those positions in sequence a low value ensuring their weights are not accounted .
'''
    return self.embedding.compute_mask(*args, **kwargs)

  # The call method is what gets run when the layer is called on input data
  def call(self, x, training=False):
    length = tf.shape(x)[1]  # Get the length of the input sequence
    length = tf.cast(length, tf.int32)
    x = self.embedding(x)  # Lookup embeddings for the input sequence (batch of token indices)
    # Scale the embeddings by the square root of the embedding dimension size
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    # Add positional encoding to the token embeddings, sliced to match the input length
    # x = x + self.pos_encoding[tf.newaxis, :length, :]
    tf.reshape(self.pos_encoding, (1, -1, self.d_model))[:, :length, :]
    x = self.dropout(x, training=training)
    '''
    The addition operation broadcasts the positional encodings across the batch, which means the same positional encoding is added to every sequence in the batch. The slicing operation self.pos_encoding[tf.newaxis, :length, :] is used to match the time axis of the positional encodings with the time axis of the input embeddings.
    '''
    return x  # Return the token embeddings combined with positional encoding

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
  def serve(self, x):
    # Replicate the logic from the call method, simplified for serving if necessary
    return self(x, training=False)
  