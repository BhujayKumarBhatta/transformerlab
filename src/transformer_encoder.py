import tensorflow as tf
# tf.random.set_seed(42)
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, RepeatVector,
    Lambda, Layer, LayerNormalization, Dropout )

from positional_encoding import PositionalEmbedding
from encoder_layer import  EncoderLayerV2





class TransformerEncoderV3(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
                 max_pos=2048, epsilon=1e-6,
                 pos_dropout=0.1,
                 en_dropout=0.1,
                 use_masked_softmax=False,
                 row_mask_to_sero=False,
                 debug=False,
                 **kwargs):
        super(TransformerEncoderV3, self).__init__(**kwargs)
        self.debug = debug
        # self.en_dropout= en_dropout
        self.num_layers = num_layers
        self.num_heads = num_heads
        # Initialize the PositionalEmbedding layer
        self.positional_embedding = PositionalEmbedding(vocab_size, d_model, max_pos,
                                                        pos_dropout=pos_dropout,
                                                      )
        self.enc_layers_0 = EncoderLayerV2(d_model, num_heads, dff, en_dropout, epsilon, 
                                           use_masked_softmax=use_masked_softmax,
                                              row_mask_to_sero=row_mask_to_sero,
                                           debug=debug)
        if num_layers > 1:
            self.remaining_layers = num_layers - 1
            self.enc_layers = [EncoderLayerV2(d_model, num_heads, dff, en_dropout, epsilon, 
                                              use_masked_softmax=use_masked_softmax,
                                              row_mask_to_sero=row_mask_to_sero,
                                              debug=False ) for _ in range(self.remaining_layers)]
        self.supports_masking = True  # Declare that this layer supports masking
        self.attention_weights = None    


    def call(self, x, training, return_attention_weights=False):
        attention_weights = []
        # Apply positional embedding
        x = self.positional_embedding(x, training=training)        
        if self.debug: print('num_heads at encoder call :', self.num_heads)
        
        x = self.enc_layers_0(x, training=training, num_heads=self.num_heads)
        layer_attention_weights = self.enc_layers_0.attention_weights
        self.debug = False
        for i in range(self.remaining_layers):
            x = self.enc_layers[i](x, training=training, num_heads=self.num_heads)
            layer_attention_weights  = self.enc_layers[i].attention_weights
            if return_attention_weights:
                attention_weights.append(layer_attention_weights)
        
        self.attention_weights = attention_weights
        return x





class TransformerEncoderV2(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
                 max_pos=2048, epsilon=1e-6,
                 pos_dropout=0.1,
                 en_dropout=0.1,
                 debug=False,
                 **kwargs):
        super(TransformerEncoderV2, self).__init__(**kwargs)
        self.debug = debug
        # self.en_dropout= en_dropout
        self.num_layers = num_layers
        self.num_heads = num_heads
        # Initialize the PositionalEmbedding layer
        self.positional_embedding = PositionalEmbedding(vocab_size, d_model, max_pos,
                                                        pos_dropout=pos_dropout,
                                                      )
        self.enc_layers_0 = EncoderLayerV2(d_model, num_heads, dff, en_dropout, epsilon, debug=debug)
        if num_layers > 1:
            self.remaining_layers = num_layers - 1
            self.enc_layers = [EncoderLayerV2(d_model, num_heads, dff, en_dropout, epsilon, debug=False ) for _ in range(self.remaining_layers)]
        self.supports_masking = True  # Declare that this layer supports masking
        self.attention_weights = None

    def convert_padding_mask_to_matrix(self, padding_masks_org):
        boolean_colvec_repeat = padding_masks_org[:, :, None],
        org_mask_num = tf.cast(padding_masks_org, dtype=tf.float32)
        shape = tf.shape(org_mask_num)
        batch_size, seqlen = shape[0], shape[1]
        # batch_size, seqlen = tf.shape(org_mask_num)
        masknum_rowvec_repeat = org_mask_num[:, None, :],
        I_matrix = tf.zeros((batch_size, seqlen, seqlen), dtype=tf.float32)
        ## idetntify where   mask will be taken as row and where rows from identity matrix will be taken .
        expanded_mask = tf.where(
              boolean_colvec_repeat, masknum_rowvec_repeat, I_matrix)
        if len(expanded_mask.shape) > 3:
            if self.debug: print('squeezing extra dimension:', expanded_mask.shape)
            expanded_mask = tf.squeeze(expanded_mask, axis=0)
        if self.debug:
            print('Expanded mask shape:', expanded_mask.shape)
            print('1st padding mask matrix:\n',expanded_mask[0])
        return expanded_mask


    def call(self, x, training, return_attention_weights=False):
        attention_weights = []
        # Apply positional embedding
        x = self.positional_embedding(x, training=training)
        encoder_mask = x._keras_mask
        encoder_mask_matrix = self.convert_padding_mask_to_matrix(encoder_mask)
        encoder_mask_matrix = tf.cast(encoder_mask_matrix > 0, dtype=tf.bool)
        if self.debug and encoder_mask is not None: print(f'encoder_mask_matrix from encoder: {encoder_mask_matrix[0]}')
        x = self.enc_layers_0(x, training=training, mask=encoder_mask_matrix)
        layer_attention_weights = self.enc_layers_0.attention_weights
        for i in range(self.remaining_layers):
            x = self.enc_layers[i](x, training=training, mask=encoder_mask_matrix)
            layer_attention_weights  = self.enc_layers[i].attention_weights
            if return_attention_weights:
                attention_weights.append(layer_attention_weights)
        self.debug = False
        self.attention_weights = attention_weights
        return x, encoder_mask











class TransformerEncoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
                 max_pos=2048, epsilon=1e-6,
                 pos_dropout=0.1,
                 en_dropout=0.1,
                 debug=False,
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.debug = debug
        # self.en_dropout= en_dropout
        self.num_layers = num_layers
        # Initialize the PositionalEmbedding layer
        self.positional_embedding = PositionalEmbedding(vocab_size, d_model, max_pos,
                                                        pos_dropout=pos_dropout,
                                                      )
        self.enc_layers_0 = EncoderLayer(d_model, num_heads, dff, en_dropout, epsilon, debug=debug)
        if num_layers > 1:
            self.remaining_layers = num_layers - 1
            self.enc_layers = [EncoderLayer(d_model, num_heads, dff, en_dropout, epsilon, debug=False ) for _ in range(self.remaining_layers)]
        self.supports_masking = True  # Declare that this layer supports masking


    def convert_padding_mask_to_matrix(self, padding_masks_org):
        boolean_colvec_repeat = padding_masks_org[:, :, None],
        org_mask_num = tf.cast(padding_masks_org, dtype=tf.float32)
        shape = tf.shape(org_mask_num)
        batch_size, seqlen = shape[0], shape[1]
        # batch_size, seqlen = tf.shape(org_mask_num)
        masknum_rowvec_repeat = org_mask_num[:, None, :],
        I_matrix = tf.zeros((batch_size, seqlen, seqlen), dtype=tf.float32)
        ## idetntify where   mask will be taken as row and where rows from identity matrix will be taken .
        expanded_mask = tf.where(
              boolean_colvec_repeat, masknum_rowvec_repeat, I_matrix)
        if len(expanded_mask.shape) > 3:
            if self.debug: print('squeezing extra dimension:', expanded_mask.shape)
            expanded_mask = tf.squeeze(expanded_mask, axis=0)
        if self.debug:
            print('Expanded mask shape:', expanded_mask.shape)
            print('1st padding mask matrix:\n',expanded_mask[0])
        return expanded_mask


    def call(self, x, training, return_attention_weights=False):
        attention_weights = []
        # Apply positional embedding
        x = self.positional_embedding(x, training=training)
        encoder_mask = x._keras_mask
        encoder_mask_matrix = self.convert_padding_mask_to_matrix(encoder_mask)
        encoder_mask_matrix = tf.cast(encoder_mask_matrix > 0, dtype=tf.bool)
        if self.debug and encoder_mask is not None: print(f'encoder_mask_matrix from encoder: {encoder_mask_matrix[0]}')
        x, layer_attention_weights = self.enc_layers_0(x, training=training, mask=encoder_mask_matrix)
        for i in range(self.remaining_layers):
            x, layer_attention_weights = self.enc_layers[i](x, training=training, mask=encoder_mask_matrix)
            if return_attention_weights:
                attention_weights.append(layer_attention_weights)
        self.debug = False
        if return_attention_weights:
            return x, attention_weights, encoder_mask
        else:
            return x, encoder_mask
