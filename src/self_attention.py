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



class SelfAttentionV4(Layer):
    def __init__(self, d_model, d_k, d_v, debug=False, 
                 use_masked_softmax=True,
                 row_mask_to_sero=False,
                 **kwargs):
        super(SelfAttentionV4, self).__init__(**kwargs)
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        # Create dense layers for Q, K, V with appropriate dimensions
        self.query_dense = Dense(d_k)
        self.key_dense = Dense(d_k)
        self.value_dense = Dense(d_v)
        self._softmax = tf.keras.layers.Softmax()
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking
        self.attention_weights = None
        self.row_mask_to_sero = row_mask_to_sero
        self.use_masked_softmax = use_masked_softmax

    def call(self, inputs_q, inputs_k, inputs_v, 
             mask=None, 
            use_causal_mask=False, num_heads=1):
        query = self.query_dense(inputs_q)  # (batch_size, seq_len, d_k)
        key = self.key_dense(inputs_k)      # (batch_size, seq_len, d_k)
        value = self.value_dense(inputs_v)  # (batch_size, seq_len, d_v)
        ### score is simply dot product divided by sqrt of dimension
        #### in the keras multihead implementation, this dot product is done by einsum
        scores = tf.matmul(query, key, transpose_b=True)  # (batch_size, seq_len, seq_len)
        scores = scores / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        if self.debug:
            print(f'scores{scores.shape} b4 mask = q{query.shape} X kT{key.shape}' )
        # if mask is not None:
        attention_mask  = self._compute_attention_mask( 
                            inputs_q, inputs_v,
                            use_causal_mask=use_causal_mask,
                            num_heads=num_heads,
                            )        
        ## Within softmax the scores are first set to high negaitve  
        ### https://github.com/keras-team/keras/blob/v2.15.0/keras/layers/activation/softmax.py
        ### then for tiny scores the softmax becomes zero 
        if self.use_masked_softmax:
            attention_weights = self._softmax(scores, attention_mask)
        else:
            masked_score, mask_float = self.get_masked_score(scores, attention_mask)
            attention_weights = tf.nn.softmax(masked_score, axis=-1)
        if self.row_mask_to_sero:
            mask_float = tf.cast(attention_mask, tf.float32)
            attention_weights *= (mask_float)
        if self.debug: 
            # tf.print(f'attention weights after softmax: ', attention_weights)
            print('attention weights after softmax:', attention_weights.numpy())
        output = tf.matmul(attention_weights, value)  # (1, batch_size, seq_len, d_v)
        if self.debug:
            print(f'attention output{output.shape} = attention_weights{attention_weights.shape} X v{value.shape}' )        
        self.attention_weights = attention_weights
        self.debug = False
        return output
   


    def _compute_attention_mask(self, query, value, 
                                key=None, 
                                attention_mask=None, 
                                use_causal_mask=False,
                                num_heads=1
                                ):
        """Computes the attention mask, using the Keras masks of the inputs.
        https://github.com/keras-team/keras/blob/v2.15.0/keras/layers/attention/multi_head_attention.py#L478
        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `use_causal_mask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        In general, if the `query` and `value` are masked, then there is no need
        to define the `attention_mask`.

        Args:
            query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
            key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
            value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions.
            use_causal_mask: A boolean to indicate whether to apply a causal
                mask to prevent tokens from attending to future tokens (e.g.,
                used in a decoder Transformer).

        Returns:
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions, based on the Keras masks of the
                `query`, `key`, `value`, and `attention_mask` tensors, and the
                causal mask if `use_causal_mask=True`.
        """
        query_mask = getattr(query, "_keras_mask", None)
        # if self.debug: print('query_mask:', query_mask)
        value_mask = getattr(value, "_keras_mask", None)
        # if self.debug: print('value_mask:', value_mask)
        key_mask = getattr(key, "_keras_mask", None)
        # if self.debug: print('key_mask:', key_mask)
        auto_mask = None
        if query_mask is not None:
            query_mask = tf.cast(query_mask, tf.bool)  # defensive casting
            # B = batch size, T = max query length
            if self.debug: print('col mask equivallent query mask:', query_mask.numpy())
            auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]
        if value_mask is not None:
            value_mask = tf.cast(value_mask, tf.bool)  # defensive casting
            # B = batch size, S == max value length
            mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            if self.debug: print('row mask equivallent value mask:', mask.numpy())
            auto_mask = mask if auto_mask is None else auto_mask & mask
            if self.debug: print('query(col) and value(row) combined mask matrix:', auto_mask.numpy()) 
        if key_mask is not None:
            key_mask = tf.cast(key_mask, tf.bool)  # defensive casting
            # B == batch size, S == max key length == max value length
            mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if use_causal_mask:
            # the shape of the causal mask is [1, T, S]
            mask = self._compute_causal_mask(query, value)
            if self.debug: print('causal mask:', mask.numpy())
            if self.debug: print('padding mask:', auto_mask.numpy())
            auto_mask = mask if auto_mask is None else auto_mask & mask
            if self.debug: print('combined mask:', auto_mask.numpy())
        if auto_mask is not None:
            # merge attention_mask & automatic mask, to shape [B, T, S]
            attention_mask = (
                auto_mask
                if attention_mask is None
                else tf.cast(attention_mask, bool) & auto_mask
            )
            if self.debug: print(f'mask b4 head  adjustment: {attention_mask.shape}')
            attention_mask = tf.repeat(attention_mask, 
                                       num_heads, axis=0)
            if self.debug: print(f'mask adjusted for {num_heads} heads: {attention_mask.shape}')
        return attention_mask



    def _compute_causal_mask(self, query, value=None):
        """Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean `Tensor` equal to:
        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]        ```

        Args:
            query: query `Tensor` of shape `(B, T, ...)`.
            value: value `Tensor` of shape `(B, S, ...)` (optional, defaults to
                query).
        Returns:
            mask: a boolean `Tensor` of shape [1, T, S] containing a lower
                triangular matrix of shape [T, S].
        """
        q_seq_length = tf.shape(query)[1]
        v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
        return tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0
        )
        


    def get_masked_score(self, raw_scores, mask):        
        mask_bool = tf.cast(mask, dtype=tf.bool)
        # Invert the boolean mask directly without casting it to float first
        ### setting False=1, means mask to be applied
        mask_inverted = tf.logical_not(mask_bool) 
        # if self.debug: print('mask inverted within selfattn:', mask_inverted[:1])
        # Now cast the inverted mask to float
        mask_float = tf.cast(mask_inverted, dtype=tf.float32)  
        if self.debug:
            tf.print(f'first original score: ', raw_scores[0])
            tf.print(f'first mask: ', mask_float[0])
            # print(f'first original score: {scores[0].numpy()}')
            # print(f'first mask: {mask_float[0].numpy()}')
        ### adding a high negative value with the score-> making the soore tiny
        masked_score = raw_scores
        masked_score += (mask_float * -1e9)  
        if self.debug: 
            print(f'first masked score: ', masked_score[0].numpy())
            # tf.print(f'first masked score: ', scores[0])
            # print(f'first masked score: {scores[0].numpy()}')
        return  masked_score, mask_float















class SelfAttentionV3(Layer):
    def __init__(self, d_model, d_k, d_v, debug=False, **kwargs):
        super(SelfAttentionV3, self).__init__(**kwargs)
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        # Create dense layers for Q, K, V with appropriate dimensions
        self.query_dense = Dense(d_k)
        self.key_dense = Dense(d_k)
        self.value_dense = Dense(d_v)
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking
        self.attention_weights = None

    def call(self, inputs_q, inputs_k, inputs_v, 
             mask=None, set_masked_row_weight_to_zero=False,
            compute_causal_mask=False):
        query = self.query_dense(inputs_q)  # (batch_size, seq_len, d_k)
        key = self.key_dense(inputs_k)      # (batch_size, seq_len, d_k)
        value = self.value_dense(inputs_v)  # (batch_size, seq_len, d_v)
        ### score is simply dot product divided by sqrt of dimension
        scores = tf.matmul(query, key, transpose_b=True)  # (batch_size, seq_len, seq_len)
        scores = scores / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        if self.debug:
            print(f'scores{scores.shape} b4 mask = q{query.shape} X kT{key.shape}' )
        if mask is not None:
            scores, mask_float = self.handle_mask(scores, mask)
        ## for tiny scores the softmax will be zero
        attention_weights = tf.nn.softmax(scores, axis=-1) 
        if mask is not None and set_masked_row_weight_to_zero:
            attention_weights *= (1 - mask_float)
        if self.debug: 
            # tf.print(f'attention weights after softmax: ', attention_weights)
            print('attention weights after softmax:', attention_weights)
        output = tf.matmul(attention_weights, value)  # (1, batch_size, seq_len, d_v)
        if self.debug:
            print(f'attention output{output.shape} = attention_weights{attention_weights.shape} X v{value.shape}' )
        self.debug = False
        self.attention_weights = attention_weights
        return output


    def convert_padding_mask_to_matrix(self, padding_mask_org):
        if self.debug: print('padding_mask_org:', padding_mask_org )
        binary_masks = tf.cast(padding_mask_org, dtype=tf.float32)
        if self.debug: print('binary_masks:', binary_masks )
        mask_row = binary_masks[:, :, tf.newaxis]
        if self.debug: print('mask_row:', mask_row )
        mask_col = binary_masks[:, tf.newaxis, :]
        if self.debug: print('mask_col:', mask_col )
        matrix_mask = tf.minimum(mask_row, mask_col) 
        if self.debug: print('matrix_mask:', matrix_mask)
        return matrix_mask 
        

    def handle_mask(self, raw_scores, mask):
        mask = self.convert_padding_mask_to_matrix(mask)
        if self.debug: print('mask received by selfattn......:', mask.shape)            
            ### without this the serve method during save_mode.save was failing
        mask_bool = tf.cast(mask, dtype=tf.bool)
        # Invert the boolean mask directly without casting it to float first
        ### setting False=1, means mask to be applied
        mask_inverted = tf.logical_not(mask_bool) 
        # if self.debug: print('mask inverted within selfattn:', mask_inverted[:1])
        # Now cast the inverted mask to float
        mask_float = tf.cast(mask_inverted, dtype=tf.float32)        
        ### when encoder_out and decoder_input have different seq_len
        if raw_scores.shape[-1] != mask.shape[-1]: 
            mask_float = tf.reshape(mask_float, 
                                    [mask_float.shape[0], mask_float.shape[-1], 1])
            if self.debug: print(
                'reshape num casted mask to match score :', mask_float.shape)
        if self.debug:
            tf.print(f'first original score: ', raw_scores[0])
            tf.print(f'first mask: ', mask_float[0])
            # print(f'first original score: {scores[0].numpy()}')
            # print(f'first mask: {mask_float[0].numpy()}')
        ### adding a high negative value with the score-> making the soore tiny
        masked_score = raw_scores
        masked_score += (mask_float * -1e9)  
        if self.debug: 
            print(f'first masked score: ', masked_score[0].numpy())
            # tf.print(f'first masked score: ', scores[0])
            # print(f'first masked score: {scores[0].numpy()}')
        return  masked_score, mask_float


    




class SelfAttentionV2(Layer):
    def __init__(self, d_model, d_k, d_v, debug=False, **kwargs):
        super(SelfAttentionV2, self).__init__(**kwargs)
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        # Create dense layers for Q, K, V with appropriate dimensions
        self.query_dense = Dense(d_k)
        self.key_dense = Dense(d_k)
        self.value_dense = Dense(d_v)
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking
        self.attention_weights = None

    def call(self, inputs_q, inputs_k, inputs_v, mask=None):
        query = self.query_dense(inputs_q)  # (batch_size, seq_len, d_k)
        key = self.key_dense(inputs_k)      # (batch_size, seq_len, d_k)
        value = self.value_dense(inputs_v)  # (batch_size, seq_len, d_v)
        ### score is simply dot product divided by sqrt of dimension
        scores = tf.matmul(query, key, transpose_b=True)  # (batch_size, seq_len, seq_len)
        scores = scores / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        if self.debug:
            print(f'scores{scores.shape} b4 mask = q{query.shape} X kT{key.shape}' )
        if mask is not None:
            if self.debug: print('mask received by selfattn......:', mask.shape)
            # Invert the boolean mask directly without casting it to float first
            mask_bool = tf.cast(mask, dtype=tf.bool)  ### without this the serve method during save_mode.save was failing
            mask_inverted = tf.logical_not(mask_bool) ### setting False=1, means mask to be applied
            # if self.debug: print('mask inverted within selfattn:', mask_inverted[:1])
            # Now cast the inverted mask to float
            mask_float = tf.cast(mask_inverted, dtype=tf.float32)
            if scores.shape[-1] != mask.shape[-1]: ### when encoder_out and decoder_input have different seq_len
                mask_float = tf.reshape(mask_float, [mask_float.shape[0], mask_float.shape[-1], 1])
                if self.debug: print('reshape num casted mask to match score :', mask_float.shape)
            if self.debug:
                tf.print(f'first original score: ', scores[0])
                tf.print(f'first mask: ', mask_float[0])
                # print(f'first original score: {scores[0].numpy()}')
                # print(f'first mask: {mask_float[0].numpy()}')
            scores += (mask_float * -1e9)  ### adding a high negative value with the score-> making the soore tiny
            if self.debug: 
                tf.print(f'first masked score: ', scores[0])
                # print(f'first masked score: {scores[0].numpy()}')
        attention_weights = tf.nn.softmax(scores, axis=-1) ## for tiny scores the softmax will be zero
        if mask is not None:
            attention_weights *= (1 - mask_float)
        if self.debug: 
            tf.print(f'attention weights after softmax: ', attention_weights)
            # print('attention weights after softmax:', attention_weights)
        output = tf.matmul(attention_weights, value)  # (1, batch_size, seq_len, d_v)
        if self.debug:
            print(f'attention output{output.shape} = attention_weights{attention_weights.shape} X v{value.shape}' )
        self.debug = False
        self.attention_weights = attention_weights
        return output
        

    # def get_serve_function(self):
    #     input_signature = [
    #         tf.TensorSpec(shape=[None, None, self.d_model], dtype=tf.float32, name="query"),
    #         tf.TensorSpec(shape=[None, None, self.d_model], dtype=tf.float32, name="key"),
    #         tf.TensorSpec(shape=[None, None, self.d_model], dtype=tf.float32, name="value"),
    #         tf.TensorSpec(shape=[None, None, None], dtype=tf.bool, name="mask"),
    #     ]
    #     @tf.function(input_signature=input_signature)
    #     def serve(self, query, key, value, mask=None):
    #         output, attention_weights = self(query, key, value, mask)
    #         return {'output': output, 'attention_weights': attention_weights}
    #     return serve

    
     
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="query"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="key"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="value"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.bool, name="mask")
        ])
    def serve(self, query, key, value, mask=None):
        # You can directly call self() which triggers the call method
        output, attention_weights = self(query, key, value, mask)
        return output












class SelfAttention(Layer):
    def __init__(self, d_model, d_k, d_v, debug=False, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        # Create dense layers for Q, K, V with appropriate dimensions
        self.query_dense = Dense(d_k)
        self.key_dense = Dense(d_k)
        self.value_dense = Dense(d_v)
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking

    def call(self, inputs_q, inputs_k, inputs_v, mask=None):
        query = self.query_dense(inputs_q)  # (batch_size, seq_len, d_k)
        key = self.key_dense(inputs_k)      # (batch_size, seq_len, d_k)
        value = self.value_dense(inputs_v)  # (batch_size, seq_len, d_v)
        ### score is simply dot product divided by sqrt of dimension
        scores = tf.matmul(query, key, transpose_b=True)  # (batch_size, seq_len, seq_len)
        scores = scores / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        if self.debug:
            print(f'scores{scores.shape} b4 mask = q{query.shape} X kT{key.shape}' )
        if mask is not None:
            if self.debug: print('mask received by selfattn......:', mask.shape)
            # Invert the boolean mask directly without casting it to float first
            mask_bool = tf.cast(mask, dtype=tf.bool)  ### without this the serve method during save_mode.save was failing
            mask_inverted = tf.logical_not(mask_bool) ### setting False=1, means mask to be applied
            # if self.debug: print('mask inverted within selfattn:', mask_inverted[:1])
            # Now cast the inverted mask to float
            mask_float = tf.cast(mask_inverted, dtype=tf.float32)
            if scores.shape[-1] != mask.shape[-1]: ### when encoder_out and decoder_input have different seq_len
                mask_float = tf.reshape(mask_float, [mask_float.shape[0], mask_float.shape[-1], 1])
                if self.debug: print('reshape num casted mask to match score :', mask_float.shape)
            if self.debug:
                tf.print(f'first original score: ', scores[0])
                tf.print(f'first mask: ', mask_float[0])
                # print(f'first original score: {scores[0].numpy()}')
                # print(f'first mask: {mask_float[0].numpy()}')
            scores += (mask_float * -1e9)  ### adding a high negative value with the score-> making the soore tiny
            if self.debug: 
                tf.print(f'first masked score: ', scores[0])
                # print(f'first masked score: {scores[0].numpy()}')
        attention_weights = tf.nn.softmax(scores, axis=-1) ## for tiny scores the softmax will be zero
        if mask is not None:
            attention_weights *= (1 - mask_float)
        if self.debug: 
            tf.print(f'attention weights after softmax: ', attention_weights)
            # print('attention weights after softmax:', attention_weights)
        output = tf.matmul(attention_weights, value)  # (1, batch_size, seq_len, d_v)
        if self.debug:
            print(f'attention output{output.shape} = attention_weights{attention_weights.shape} X v{value.shape}' )
        self.debug = False
        return output, attention_weights
        

    # def get_serve_function(self):
    #     input_signature = [
    #         tf.TensorSpec(shape=[None, None, self.d_model], dtype=tf.float32, name="query"),
    #         tf.TensorSpec(shape=[None, None, self.d_model], dtype=tf.float32, name="key"),
    #         tf.TensorSpec(shape=[None, None, self.d_model], dtype=tf.float32, name="value"),
    #         tf.TensorSpec(shape=[None, None, None], dtype=tf.bool, name="mask"),
    #     ]
    #     @tf.function(input_signature=input_signature)
    #     def serve(self, query, key, value, mask=None):
    #         output, attention_weights = self(query, key, value, mask)
    #         return {'output': output, 'attention_weights': attention_weights}
    #     return serve

    
     
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="query"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="key"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="value"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.bool, name="mask")
        ])
    def serve(self, query, key, value, mask=None):
        # You can directly call self() which triggers the call method
        output, attention_weights = self(query, key, value, mask)
        return output




        
