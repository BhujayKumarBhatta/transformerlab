import tensorflow as tf
# tf.random.set_seed(42)
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, RepeatVector,
    Lambda, Layer, LayerNormalization, Dropout )

from mha import MultiHeadAttentionV3
from add_norm import AddNorm
from positionwise_ff import FeedForward




class EncoderLayerV5(Layer):
    ''' parameterised selection of MHA'''
    def __init__(self, d_model, num_heads, dff,
                 en_dropout=0.1,
                 epsilon=1e-6, 
                 use_masked_softmax=False,
                 row_mask_to_sero=False,
                 use_bbmha=True,
                 return_attention_scores=False,
                 debug=False,
                 **kwargs):
        super(EncoderLayerV5, self).__init__(**kwargs)
        self.debug=debug
        if use_bbmha:
            self.mha = MultiHeadAttentionV3(num_heads, d_model, 
                                            use_masked_softmax=use_masked_softmax,
                                             row_mask_to_sero=row_mask_to_sero,
                                            debug=debug)
            if self.debug: print('using bb mha')
        else:
            self.mha = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=d_model)
            if self.debug: print('using tf mha ')
            
        self.add_norm1 = AddNorm(epsilon=epsilon)
        self.add_norm2 = AddNorm(epsilon=epsilon)
        self.ffn = FeedForward(d_model, dff)
        self.en_dropout = en_dropout
        self.dropout1 = Dropout(en_dropout)
        self.dropout2 = Dropout(en_dropout)        
        self.supports_masking = True  # Declare that this layer supports masking
        self.attention_weights = None
        self.use_bbmha = use_bbmha
        

    def call(self, x, training=False, num_heads=1, mask=None):
        if self.debug: print('mask received at encoder layer:', mask.shape)
        if self.use_bbmha:
            # attn_output  = self.mha(x, x, x, num_heads=self.num_heads, mask=mask)
            attn_output  = self.mha(x, x, x)
            self.attention_weights = self.mha.attention_weights
        else:            
            attn_output = self.mha(query=x, value=x, key=x, 
                                        return_attention_scores=False)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.add_norm1(x, attn_output)
        ffn_output = self.ffn(out1)
        # ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.add_norm2(out1, ffn_output)
        self.debug = False
        return out2








class EncoderLayerV2(Layer):
    def __init__(self, d_model, num_heads, dff,
                 en_dropout=0.1,
                 epsilon=1e-6, debug=False, 
                 use_masked_softmax=False,
                 row_mask_to_sero=False,
                 **kwargs):
        super(EncoderLayerV2, self).__init__(**kwargs)
        self.mha = MultiHeadAttentionV3(num_heads, d_model, 
                                        use_masked_softmax=use_masked_softmax,
                                         row_mask_to_sero=row_mask_to_sero,
                                        debug=debug)
        self.add_norm1 = AddNorm(epsilon=epsilon)
        self.add_norm2 = AddNorm(epsilon=epsilon)
        self.ffn = FeedForward(d_model, dff)
        self.en_dropout = en_dropout
        self.dropout1 = Dropout(en_dropout)
        self.dropout2 = Dropout(en_dropout)
        self.debug=debug
        self.supports_masking = True  # Declare that this layer supports masking
        self.attention_weights = None

    def call(self, x, training, num_heads=1, mask=None):
        if self.debug: print('mask received at encoder layer:', mask.shape)
        attn_output  = self.mha(x, x, x, num_heads=num_heads, mask=mask)  # Assuming you have implemented MultiHeadAttention        
        self.attention_weights = self.mha.attention_weights
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.add_norm1(x, attn_output)
        ffn_output = self.ffn(out1)
        # ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.add_norm2(out1, ffn_output)
        self.debug = False
        return out2




class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff,
                 en_dropout=0.1,
                 epsilon=1e-6, debug=False, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads, d_model, debug=debug)
        self.add_norm1 = AddNorm(epsilon=epsilon)
        self.add_norm2 = AddNorm(epsilon=epsilon)
        self.ffn = FeedForward(d_model, dff)
        self.en_dropout = en_dropout
        self.dropout1 = Dropout(en_dropout)
        self.dropout2 = Dropout(en_dropout)
        self.debug=debug
        self.supports_masking = True  # Declare that this layer supports masking

    def call(self, x, training, mask=None):
        if self.debug: print('mask received at encoder layer:', mask.shape)
        attn_output, atten_weights = self.mha(x, x, x, mask=mask)  # Assuming you have implemented MultiHeadAttention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.add_norm1(x, attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.add_norm2(out1, ffn_output)
        self.debug = False
        return out2, atten_weights


    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="x"),
        tf.TensorSpec(shape=[], dtype=tf.bool, name="training"),
        tf.TensorSpec(shape=[None, None], dtype=tf.bool, name="mask")])
    def serve(self, x, training=False, mask=None):
        # Mirror the logic of the call method, setting training to False for serving
        attn_output, atten_weights = self.mha(x, x, x, mask=mask)
        # attn_output = self.dropout1(attn_output, training=training)
        out1 = self.add_norm1(x, attn_output)
        ffn_output = self.ffn(out1)
        # ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.add_norm2(out1, ffn_output)
        return {'output': out2, 'attention_weights': atten_weights}


