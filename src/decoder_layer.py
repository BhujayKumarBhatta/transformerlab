import tensorflow as tf
# tf.random.set_seed(42)
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, RepeatVector,
    Lambda, Layer, LayerNormalization, Dropout )

from mha import MultiHeadAttention, MultiHeadAttentionV3
from add_norm import AddNorm
from positionwise_ff import FeedForward


class DecoderLayerV3(Layer):
    def __init__(self, d_model, num_heads, dff, epsilon=1e-6,
                 de_dropout=01., 
                 use_masked_softmax=False,
                 row_mask_to_sero=False,
                 debug=False, **kwargs):
        super(DecoderLayerV3, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.mha1 = MultiHeadAttentionV3(num_heads, d_model, 
                                         use_masked_softmax=use_masked_softmax,
                                         row_mask_to_sero=row_mask_to_sero,
                                         debug=debug)  # Masked self-attention
        self.mha2 = MultiHeadAttentionV3(num_heads, d_model, 
                                         use_masked_softmax=use_masked_softmax,
                                         row_mask_to_sero=row_mask_to_sero,
                                         debug=debug)  # Encoder-decoder attention
        self.ffn = FeedForward(d_model, dff)
        self.add_norm1 = AddNorm(epsilon=epsilon)
        self.add_norm2 = AddNorm(epsilon=epsilon)
        self.add_norm3 = AddNorm(epsilon=epsilon)
        self.dropout1 = tf.keras.layers.Dropout(de_dropout)
        self.dropout2 = tf.keras.layers.Dropout(de_dropout)
        self.dropout3 = tf.keras.layers.Dropout(de_dropout)
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking

    def call(self, x, enc_output, training, use_causal_mask=True):
        if self.debug: print('start decoder masked multihead :')
        attn1 = self.mha1(x, x, x,
                                               num_heads=self.num_heads,
                                               use_causal_mask=use_causal_mask
                                              )
        attn_weights_block1 = self.mha1.attention_weights
        attn1 = self.dropout1(attn1, training=training)
        if self.debug: print('attn1: ', attn1.shape)
        out1 = self.add_norm1(x, attn1)
        if self.debug: print('out1: ', out1.shape)
        if self.debug: print('decoder masked multihead done.\n')
        # Encoder-decoder attention
        if self.debug: print('starting multihead for encoder_out: ')
        ### provide the encoder_mask
        # if self.debug: print('encoder_mask reshaped by decoder:', encoder_mask)
        attn2 = self.mha2(out1, enc_output, enc_output, 
                                               num_heads=self.num_heads,
                                               use_causal_mask=use_causal_mask)
        attn_weights_block2 = self.mha2.attention_weights
        attn2 = self.dropout2(attn2, training=training)
        if self.debug: print('attn2: ', attn2.shape)
        out2 = self.add_norm2(out1, attn2)
        if self.debug: print('out2: ', out2.shape)
        # Feedforward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.add_norm3(out2, ffn_output)
        self.debug = False
        return out3, attn_weights_block1, attn_weights_block2
















class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, epsilon=1e-6,
                 de_dropout=01., debug=False, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)        
        self.mha1 = MultiHeadAttention(num_heads, d_model, debug=debug)  # Masked self-attention
        self.mha2 = MultiHeadAttention(num_heads, d_model, debug=debug)  # Encoder-decoder attention
        self.ffn = FeedForward(d_model, dff)
        self.add_norm1 = AddNorm(epsilon=epsilon)
        self.add_norm2 = AddNorm(epsilon=epsilon)
        self.add_norm3 = AddNorm(epsilon=epsilon)
        self.dropout1 = tf.keras.layers.Dropout(de_dropout)
        self.dropout2 = tf.keras.layers.Dropout(de_dropout)
        self.dropout3 = tf.keras.layers.Dropout(de_dropout)
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking

    def call(self, x, enc_output, encoder_mask, look_ahead_mask, padding_mask, training):
        if self.debug: print('start decoder masked multihead :')
        attn1, attn_weights_block1 = self.mha1(x, x, x, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        if self.debug: print('attn1: ', attn1.shape)
        out1 = self.add_norm1(x, attn1)
        if self.debug: print('out1: ', out1.shape)
        if self.debug: print('decoder masked multihead done.\n')
        # Encoder-decoder attention
        if self.debug: print('starting multihead for encoder_out: ')
        ### provide the encoder_mask
        if self.debug: print('encoder_mask reshaped by decoder:', encoder_mask)
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, mask=encoder_mask)
        attn2 = self.dropout2(attn2, training=training)
        if self.debug: print('attn2: ', attn2.shape)
        out2 = self.add_norm2(out1, attn2)
        if self.debug: print('out2: ', out2.shape)
        # Feedforward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.add_norm3(out2, ffn_output)
        self.debug = False
        return out3, attn_weights_block1, attn_weights_block2
# decoder_layer = DecoderLayer(d_model, num_heads, dff)
# decoder_layer_out = decoder_layer(embedded_data, encoder_out, encoder_out)
# print('decoder_layer_out:', decoder_layer_out)





