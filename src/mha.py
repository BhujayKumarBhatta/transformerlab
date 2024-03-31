

import pathlib
import os
import re
import string
from unicodedata import normalize
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

import warnings

# Suppress TensorFlow logging (1=filter INFO, 2=filter INFO and WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress Python warnings
warnings.filterwarnings('ignore')

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

# tf.random.set_seed(42)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Input, Dense,
     Concatenate, RepeatVector,
    Lambda, Layer, Embedding,  Dot, LayerNormalization, Dropout)

from self_attention import SelfAttentionV4



class MultiHeadAttentionV3(Layer):
    def __init__(self, num_heads, d_model, 
                 use_masked_softmax=True,
                 row_mask_to_sero=False,
                 debug=False, **kwargs):
        super(MultiHeadAttentionV3, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads
        self.Wq = Dense(d_model)
        self.Wk = Dense(d_model)
        self.Wv = Dense(d_model)
        self.final_dense = Dense(d_model)
        self.self_attention = SelfAttentionV4(d_model, self.depth, self.depth, 
                                              use_masked_softmax=use_masked_softmax,
                                              row_mask_to_sero=row_mask_to_sero,
                                              debug=debug)
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking
        self.attention_weights = None
        self.use_masked_softmax=use_masked_softmax,
        self.row_mask_to_sero=row_mask_to_sero


    def split_heads(self, x, data_type='data'):
        shape = tf.shape(x)
        batch_size, seq_len = shape[0], shape[1]
        # batch_size = x.shape[0]
        # seq_len = x.shape[1]
        """Split the last dimension into (num_heads, projection outcome(=depth))."""
        if self.debug: print(f'{data_type} b4 split head: {x.shape}')
        splitout = tf.reshape(x, (batch_size, self.num_heads, seq_len, -1))
        if self.debug: print(f'{data_type} after split head: {splitout.shape}')
        # splitout_transpose = tf.transpose(splitout, perm=[0, 2, 1, 3])
        # if self.debug: print('after tranpose split:', splitout_transpose.shape)
        return splitout

    def call(self, xq, xk, xv, mask=None, use_causal_mask=False, num_heads=1):
        batch_size = tf.shape(xq)[0]
        xq_mask = getattr(xq, "_keras_mask", None)
        xk_mask = getattr(xk, "_keras_mask", None)
        xv_mask = getattr(xv, "_keras_mask", None)
        # print('xq_mask:', xq_mask.numpy())
        # Linear projections in batch from d_model => h * d_k
        q = self.split_heads(self.Wq(xq), data_type='q')  # (batch_size, num_heads, seq_len, depth)
        # print('q  mask  after split projection:', getattr(q, "_keras_mask", None))
        k = self.split_heads(self.Wk(xk), data_type='k')
        v = self.split_heads(self.Wv(xv), data_type='v')
        # Reshape for passing through self-attention
        q = tf.reshape(q, (-1, tf.shape(q)[2], self.depth))  # (batch_size*num_heads, seq_len, depth)
        if self.debug: print('q preprared for selfattention(batch_cross_numheads):', q.shape)
        k = tf.reshape(k, (-1, tf.shape(k)[2], self.depth))
        if self.debug: print('k, v preprared for selfattention(batch_cross_numheads):', k.shape)
        v = tf.reshape(v, (-1, tf.shape(v)[2], self.depth))        
        # re insert the mask attribute and Apply self-attention to the projected vectors for q, k, v
        q._keras_mask = xq_mask
        k._keras_mask = xk_mask
        v._keras_mask = xv_mask        
        attention_output = self.self_attention(q, k, v, mask=mask,
                                               use_causal_mask=use_causal_mask,
                                              num_heads=num_heads)
        attention_weights = self.self_attention.attention_weights
        # if self.debug: print('attention_output:', attention_output.shape)
        # Reshape attention_output back to the original multi-head shape for concatenation
        attention_output_reshaped = tf.reshape(attention_output, (batch_size, self.num_heads, -1, self.depth))
        if self.debug: print('attention_output_transpose:', attention_output_reshaped.shape)
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model)) 
        if self.debug: print('concat_attention:', concat_attention.shape)
        output = self.final_dense(concat_attention)
        self.debug = False
        self.attention_weights = attention_weights
        return output

    
        




class MultiHeadAttentionV2(Layer):
    def __init__(self, num_heads, d_model, debug=False, **kwargs):
        super(MultiHeadAttentionV2, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads
        self.Wq = Dense(d_model)
        self.Wk = Dense(d_model)
        self.Wv = Dense(d_model)
        self.final_dense = Dense(d_model)
        self.self_attention = SelfAttentionV(d_model, self.depth, self.depth, debug=debug)
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking
        self.attention_weights = None


    def split_heads(self, x, data_type='data'):
        shape = tf.shape(x)
        batch_size, seq_len = shape[0], shape[1]
        # batch_size = x.shape[0]
        # seq_len = x.shape[1]
        """Split the last dimension into (num_heads, projection outcome(=depth))."""
        if self.debug: print(f'{data_type} b4 split head: {x.shape}')
        splitout = tf.reshape(x, (batch_size, self.num_heads, seq_len, -1))
        if self.debug: print(f'{data_type} after split head: {splitout.shape}')
        # splitout_transpose = tf.transpose(splitout, perm=[0, 2, 1, 3])
        # if self.debug: print('after tranpose split:', splitout_transpose.shape)
        return splitout

    def call(self, xq, xk, xv, mask=None):
        batch_size = tf.shape(xq)[0]
        # Linear projections in batch from d_model => h * d_k
        q = self.split_heads(self.Wq(xq), data_type='q')  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(self.Wk(xk), data_type='k')
        v = self.split_heads(self.Wv(xv), data_type='v')
        # Reshape for passing through self-attention
        q = tf.reshape(q, (-1, tf.shape(q)[2], self.depth))  # (batch_size*num_heads, seq_len, depth)
        if self.debug: print('q preprared for selfattention(batch_cross_numheads):', q.shape)
        k = tf.reshape(k, (-1, tf.shape(k)[2], self.depth))
        if self.debug: print('k, v preprared for selfattention(batch_cross_numheads):', k.shape)
        v = tf.reshape(v, (-1, tf.shape(v)[2], self.depth))
        if mask is not None:
            if self.debug: print('mask b4 adjusted for matrix and heads by mha:', mask.shape)
            if len(mask.shape) == 2: ### For padding mask ( batch, seqlen)
                mask = mask[:, tf.newaxis, :]  ### (batch, 1, seqlen) within selfattention score is a matrix
            mask = tf.repeat(mask, self.num_heads, axis=0)  # Adjust mask for number of heads (8, 1, seqlen) for padding_mask or (8, seqlen, seqlen) for combined_mask
            if self.debug: print('mask adjusted for matrix and heads by mha:', mask.shape)
        # Apply self-attention to the projected vectors for q, k, v
        attention_output = self.self_attention(q, k, v, mask=mask)
        attention_weights = self.self_attention.attention_weights
        # if self.debug: print('attention_output:', attention_output.shape)
        # Reshape attention_output back to the original multi-head shape for concatenation
        attention_output_reshaped = tf.reshape(attention_output, (batch_size, self.num_heads, -1, self.depth))
        if self.debug: print('attention_output_transpose:', attention_output_reshaped.shape)
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model)) ### Option 1: Direct concat using reshaping####   (1, 8, 9, 64) ==> (1, 9, 512)  ==> 8 x 64 == adding last dimension
        # splitted_attention_output = tf.split(attention_output_transpose, self.num_heads, axis=2)     ### Option 2: Concatenate all heads output
        # concat_attention = tf.concat(splitted_attention_output, axis=-1) ##(1, 9, 1, 512)
        # concat_attention = tf.squeeze(concat_attention, axis=-2) ## removing the extra one
        if self.debug: print('concat_attention:', concat_attention.shape)
        output = self.final_dense(concat_attention)
        self.debug = False
        self.attention_weights = attention_weights
        return output










class MultiHeadAttention(Layer):
    def __init__(self, num_heads, d_model, debug=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads
        self.Wq = Dense(d_model)
        self.Wk = Dense(d_model)
        self.Wv = Dense(d_model)
        self.final_dense = Dense(d_model)
        self.self_attention = SelfAttention(d_model, self.depth, self.depth, debug=debug)
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking


    def split_heads(self, x, data_type='data'):
        shape = tf.shape(x)
        batch_size, seq_len = shape[0], shape[1]
        # batch_size = x.shape[0]
        # seq_len = x.shape[1]
        """Split the last dimension into (num_heads, projection outcome(=depth))."""
        if self.debug: print(f'{data_type} b4 split head: {x.shape}')
        splitout = tf.reshape(x, (batch_size, self.num_heads, seq_len, -1))
        if self.debug: print(f'{data_type} after split head: {splitout.shape}')
        # splitout_transpose = tf.transpose(splitout, perm=[0, 2, 1, 3])
        # if self.debug: print('after tranpose split:', splitout_transpose.shape)
        return splitout

    def call(self, xq, xk, xv, mask=None):
        batch_size = tf.shape(xq)[0]
        # Linear projections in batch from d_model => h * d_k
        q = self.split_heads(self.Wq(xq), data_type='q')  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(self.Wk(xk), data_type='k')
        v = self.split_heads(self.Wv(xv), data_type='v')
        # Reshape for passing through self-attention
        q = tf.reshape(q, (-1, tf.shape(q)[2], self.depth))  # (batch_size*num_heads, seq_len, depth)
        if self.debug: print('q preprared for selfattention(batch_cross_numheads):', q.shape)
        k = tf.reshape(k, (-1, tf.shape(k)[2], self.depth))
        if self.debug: print('k, v preprared for selfattention(batch_cross_numheads):', k.shape)
        v = tf.reshape(v, (-1, tf.shape(v)[2], self.depth))
        if mask is not None:
            if self.debug: print('mask b4 adjusted for matrix and heads by mha:', mask.shape)
            if len(mask.shape) == 2: ### For padding mask ( batch, seqlen)
                mask = mask[:, tf.newaxis, :]  ### (batch, 1, seqlen) within selfattention score is a matrix
            mask = tf.repeat(mask, self.num_heads, axis=0)  # Adjust mask for number of heads (8, 1, seqlen) for padding_mask or (8, seqlen, seqlen) for combined_mask
            if self.debug: print('mask adjusted for matrix and heads by mha:', mask.shape)
        # Apply self-attention to the projected vectors for q, k, v
        attention_output, attention_weights = self.self_attention(q, k, v, mask=mask)
        # if self.debug: print('attention_output:', attention_output.shape)
        # Reshape attention_output back to the original multi-head shape for concatenation
        attention_output_reshaped = tf.reshape(attention_output, (batch_size, self.num_heads, -1, self.depth))
        if self.debug: print('attention_output_transpose:', attention_output_reshaped.shape)
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model)) ### Option 1: Direct concat using reshaping####   (1, 8, 9, 64) ==> (1, 9, 512)  ==> 8 x 64 == adding last dimension
        # splitted_attention_output = tf.split(attention_output_transpose, self.num_heads, axis=2)     ### Option 2: Concatenate all heads output
        # concat_attention = tf.concat(splitted_attention_output, axis=-1) ##(1, 9, 1, 512)
        # concat_attention = tf.squeeze(concat_attention, axis=-2) ## removing the extra one
        if self.debug: print('concat_attention:', concat_attention.shape)
        output = self.final_dense(concat_attention)
        self.debug = False
        return output, attention_weights

    # def get_serve_function(self):
    #     input_signature = [
    #         tf.TensorSpec(shape=[None, None, self.d_model], dtype=tf.float32, name="query"),
    #         tf.TensorSpec(shape=[None, None, self.d_model], dtype=tf.float32, name="key"),
    #         tf.TensorSpec(shape=[None, None, self.d_model], dtype=tf.float32, name="value"),
    #         tf.TensorSpec(shape=[None, None, None], dtype=tf.bool, name="mask", )
    #     ]

    
    #     @tf.function(input_signature=input_signature)
    #     def serve(self, query, key, value, mask=None):
    #         output, attention_weights = self(query, key, value, mask=None)
    #         return {'output': output, 'attention_weights': attention_weights}
            
    #     return serve


    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="query"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="key"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="value"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.bool, name="mask", )
        ])
    def serve(self, query, key, value, mask=None):
        output, attention_weights = self(query, key, value, mask=None)
        return {'output': output, 'attention_weights': attention_weights}
        
    
    