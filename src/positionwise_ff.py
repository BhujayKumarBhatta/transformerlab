import tensorflow as tf
# tf.random.set_seed(42)
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, RepeatVector,
    Lambda, Layer, LayerNormalization, Dropout )


class FeedForward(Layer):
    def __init__(self, d_model, dff, rate=0.1, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        # dff: dimensionality of the inner layer
        self.dense1 = Dense(dff, activation='relu') ## First Linear transformation with ReLU
        self.dense2 = Dense(d_model) ## Second Linear Transformation
        self.dropout = Dropout(rate)
        self.supports_masking = True  # Declare that this layer supports masking

    def call(self, inputs, training=True):
        x = self.dense1(inputs) ## the output is with activation
        x = self.dropout(x, training=training) # activation inbetween first and 2nd linear transformation
        x = self.dense2(x)
        return x


    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="inputs")])
    def serve(self, inputs):
        # Use the same logic as the call method, with training set to False for serving
        return {'output': self(inputs, training=False)}