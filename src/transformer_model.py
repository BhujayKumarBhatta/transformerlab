import tensorflow as tf
# tf.random.set_seed(42)
from decoder_layer import  DecoderLayer
from transformer_encoder import TransformerEncoderV2, TransformerEncoderV3
from transformer_decoder import TransformerDecoder, TransformerDecoderV3




class TransformerModelV3(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size,  target_vocab_size,
                 pe_input, pe_target, pos_dropout=0.1,
                 en_dropout=0.1, de_dropout=0.1,
                 use_masked_softmax=False,
                 row_mask_to_sero=False,
                 debug=False, **kwargs):
        super(TransformerModelV3, self).__init__()
        self.num_heads = num_heads
        self.encoder = TransformerEncoderV3(num_layers, d_model, num_heads, dff,
                                          input_vocab_size, max_pos=pe_input,
                                         pos_dropout=pos_dropout,
                                         en_dropout=en_dropout,
                                           use_masked_softmax=use_masked_softmax,
                                              row_mask_to_sero=row_mask_to_sero)
        self.decoder = TransformerDecoderV3(num_layers, d_model, num_heads, dff,
                                          target_vocab_size, max_pos_encoding=pe_target,
                                          pos_dropout=pos_dropout,
                                          de_dropout=de_dropout,
                                        use_masked_softmax=use_masked_softmax,
                                              row_mask_to_sero=row_mask_to_sero,
                                          debug=debug)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking
        self.attention_weights = None

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training):
        inp, tar = inputs
        # encoder_mask = x._keras_mask
        # if self.debug: print(f'encoder mask: {encoder_mask}')
        enc_output = self.encoder(inp, training=training)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output,
                                                     training=training)
        self.attention_weights = attention_weights
        logits = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        self.debug = False
        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass
        return logits

    def build(self, input_shape):
      input_shape = list(input_shape)
      super(TransformerModelV3, self).build(input_shape)


















class TransformerModelV2(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size,  target_vocab_size,
                 pe_input, pe_target, pos_dropout=0.1,
                 en_dropout=0.1, de_dropout=0.1,
                 debug=False, **kwargs):
        super(TransformerModelV2, self).__init__()
        self.encoder = TransformerEncoderV2(num_layers, d_model, num_heads, dff,
                                          input_vocab_size, max_pos=pe_input,
                                         pos_dropout=pos_dropout,
                                         en_dropout=en_dropout)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff,
                                          target_vocab_size, max_pos_encoding=pe_target,
                                          pos_dropout=pos_dropout,
                                          de_dropout=de_dropout,
                                          debug=debug)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking
        self.attention_weights = None

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training):
        inp, tar = inputs
        # encoder_mask = x._keras_mask
        # if self.debug: print(f'encoder mask: {encoder_mask}')
        enc_output, encoder_mask = self.encoder(inp, training=training)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output,
                                                     encoder_mask, training=training)
        self.attention_weights = attention_weights
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output

    def build(self, input_shape):
      input_shape = list(input_shape)
      super(TransformerModelV2, self).build(input_shape)












class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size,  target_vocab_size,
                 pe_input, pe_target, pos_dropout=0.1,
                 en_dropout=0.1, de_dropout=0.1,
                 debug=False, **kwargs):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dff,
                                          input_vocab_size, max_pos=pe_input,
                                         pos_dropout=pos_dropout,
                                         en_dropout=en_dropout)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff,
                                          target_vocab_size, max_pos_encoding=pe_target,
                                          pos_dropout=pos_dropout,
                                          de_dropout=de_dropout,
                                          debug=debug)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.debug = debug
        self.supports_masking = True  # Declare that this layer supports masking

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training):
        inp, tar = inputs
        # encoder_mask = x._keras_mask
        # if self.debug: print(f'encoder mask: {encoder_mask}')
        enc_output, encoder_mask = self.encoder(inp, training=training)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output,
                                                     encoder_mask, training=training)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights

    def build(self, input_shape):
      input_shape = list(input_shape)
      super(TransformerModel, self).build(input_shape)


    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input'),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='target')])
    def serve(self, input, target):
        # Assume `input` is your input sequence and `target` is the starting token for decoding
        # You may need to adjust this logic based on your specific needs
        predictions = self([input, target], training=False)
        return predictions

