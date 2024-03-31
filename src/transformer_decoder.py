import tensorflow as tf
# tf.random.set_seed(42)
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, RepeatVector,
    Lambda, Layer, LayerNormalization, Dropout )

from positional_encoding import PositionalEmbedding
from decoder_layer import DecoderLayer, DecoderLayerV3





class TransformerDecoderV3(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 vocab_size, max_pos_encoding=2048, epsilon=1e-6,
                 pos_dropout=0.1,
                 de_dropout=0.1, 
                 use_masked_softmax=False,
                 row_mask_to_sero=False,
                 debug=False, **kwargs):
        super(TransformerDecoderV3, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.positional_embedding = PositionalEmbedding(vocab_size, d_model,
                                                        max_pos_encoding, pos_dropout=pos_dropout)
        self.dec_layer_1 = DecoderLayerV3(d_model, num_heads, dff, epsilon, de_dropout, 
                                          use_masked_softmax=use_masked_softmax,
                                              row_mask_to_sero=row_mask_to_sero,
                                          debug=debug)
        self.dec_layers = [DecoderLayerV3(d_model, num_heads, dff, epsilon, de_dropout,
                                         use_masked_softmax=use_masked_softmax,
                                         row_mask_to_sero=row_mask_to_sero,) for _ in range(num_layers-1)]
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.debug=debug
        self.supports_masking = True  # Declare that this layer supports masking

    def call(self, x, enc_output, mask=None, use_causal_mask=True, training=False):
        x = self.positional_embedding(x, training=training)
        if self.debug: print('mask for decoder input:', x._keras_mask)
        if hasattr(x, '_keras_mask'):
            padding_mask = x._keras_mask
        seq_len = tf.shape(x)[1]
        attention_weights = {}       
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if self.debug: print('x sqrt: ', x.shape)
        x._keras_mask = padding_mask
        x, block1, block2 =  self.dec_layer_1(x, enc_output, 
                                              # num_heads=self.num_heads,
                                              use_causal_mask=use_causal_mask,
                                             training=training)
        if self.debug: print('x first: ', x.shape)
        self.debug=False
        attention_weights[f'decoder_layer{0}_block1'] = block1
        attention_weights[f'decoder_layer{0}_block2'] = block2
        for i in range(self.num_layers - 1):
            x, block1, block2 = self.dec_layers[i](x, enc_output, 
                                                   # num_heads=self.num_heads,
                                                   use_causal_mask=use_causal_mask,
                                                  training=training)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
            
        return x, attention_weights






class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 vocab_size, max_pos_encoding=2048, epsilon=1e-6,
                 pos_dropout=0.1,
                 de_dropout=0.1, debug=False, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.positional_embedding = PositionalEmbedding(vocab_size, d_model,
                                                        max_pos_encoding, pos_dropout=pos_dropout)
        self.dec_layer_1 = DecoderLayer(d_model, num_heads, dff, epsilon, de_dropout, debug=debug)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, epsilon, de_dropout) for _ in range(num_layers-1)]
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.debug=debug
        self.supports_masking = True  # Declare that this layer supports masking


    def create_look_ahead_mask(self, size):
        ## remember masking within SelfAttention expects False(=0) for mask to be applied
        ### bandpart(input, -1, 0)  gives a lower traingular matrix
        ##the signature is  tf.linalg.band_part(input, num_lower, num_upper)
        lhd_mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return lhd_mask


    def convert_padding_mask_to_matrix(self, padding_masks_org):
        boolean_colvec_repeat = padding_masks_org[:, :, None],
        org_mask_num = tf.cast(padding_masks_org, dtype=tf.float32)
        # batch_size, seqlen = tf.shape(org_mask_num)
        shape = tf.shape(org_mask_num)
        batch_size, seqlen = shape[0], shape[1]
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


    def create_cross_mask(self, enc_mask, dec_mask):
        bool_colvec_repeat = dec_mask[:, :, None]
        enc_mask_num = tf.cast(enc_mask, tf.float32)
        enc_rowvec_repeat = enc_mask_num[:, None, :]
        en_shape = tf.shape(enc_mask)
        batch_size, enc_seqlen = en_shape[0], en_shape[1]
        dec_shape = tf.shape(dec_mask)
        batch_size, dec_seqlen = dec_shape[0], dec_shape[1]
        zero_matrix = tf.zeros((batch_size, dec_seqlen, enc_seqlen), dtype=tf.float32)
        cross_mask = tf.where(bool_colvec_repeat, enc_rowvec_repeat, zero_matrix)
        if len(cross_mask.shape) > 3:
            if self.debug: print('squeezing extra dimension:', cross_mask.shape)
            cross_mask = tf.squeeze(cross_mask, axis=0)
        cross_mask = tf.cast(cross_mask > 0, dtype=tf.bool)
        if self.debug:
            print('cross_mask shape:', cross_mask.shape)
            print('cross_mask:', cross_mask)
        return cross_mask


    def combine_masks(self, padding_mask, look_ahead_mask):
        if self.debug: print('look_ahead_mask:', look_ahead_mask.shape)
        if self.debug: print('padding_mask:', padding_mask.shape)
        # look_ahead_mask is [seq_len, seq_len] and padding_mask is [batch_size, seq_len]
        if padding_mask is not None:
            # padding_mask_reshaped = padding_mask[:, tf.newaxis, :]
            padding_mask_reshaped = self.convert_padding_mask_to_matrix(padding_mask)
            if self.debug: print('padding_mask_reshaped:', padding_mask_reshaped.shape)
            lhd_mask_reshaped = look_ahead_mask[tf.newaxis, :, :]
            if self.debug: print('lhd_mask_reshaped:', lhd_mask_reshaped.shape)
            padding_mask_reshaped_num = tf.cast(padding_mask_reshaped, dtype=tf.float32)  # Ensure float32 for calculations
            combined_mask = tf.minimum(padding_mask_reshaped_num, lhd_mask_reshaped)
        else:
            # If no padding mask, prepare look_ahead_mask for broadcasting across batch and num_heads
            combined_mask = look_ahead_mask[tf.newaxis, :, :]
        # Ensure boolean type for combined mask
        combined_mask = tf.cast(combined_mask > 0, dtype=tf.bool)
        # combined_mask = combined_mask[tf.newaxis, :, :]   ###[1, seq_len, seq_len]
        if self.debug: print('first example bool combined_mask:\n', combined_mask[0])
        return combined_mask


    def call(self, x, enc_output, encoder_mask, training, mask=None):
        x = self.positional_embedding(x, training=training)
        if hasattr(x, '_keras_mask'):
            padding_mask = x._keras_mask
        else:
            padding_mask = None
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        look_ahead_mask = self.create_look_ahead_mask(seq_len)
        combined_mask = self.combine_masks(padding_mask, look_ahead_mask)
        cross_mask = self.create_cross_mask(encoder_mask, padding_mask)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if self.debug: print('x sqrt: ', x.shape)
        x, block1, block2 =  self.dec_layer_1(x, enc_output, cross_mask,
                                              combined_mask, padding_mask,
                                             training=training)
        if self.debug: print('x first: ', x.shape)
        attention_weights[f'decoder_layer{0}_block1'] = block1
        attention_weights[f'decoder_layer{0}_block2'] = block2
        for i in range(self.num_layers - 1):
            x, block1, block2 = self.dec_layers[i](x, enc_output, cross_mask,
                                                   combined_mask, padding_mask,
                                                  training=training)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
            self.debug=False
        return x, attention_weights
