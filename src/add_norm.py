import tensorflow as tf
# tf.random.set_seed(42)
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, RepeatVector,
    Lambda, Layer, LayerNormalization, Dropout )

class AddNorm(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.supports_masking = True  # Declare that this layer supports masking
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
    
    def build(self, input_shape):
        self.norm = LayerNormalization(epsilon=self.epsilon)
        super(AddNorm, self).build(input_shape)
    

    def call(self, inputs, sublayer_output):
        # inputs: output from the previous layer
        # sublayer_output: output from the self-attention or feedforward network
        x = self.add([inputs, sublayer_output])
        x = self.norm(x)       
        return x


    def get_dynamic_serve_function(self):
        def serve_dynamic(inputs, sublayer_output):
            # Directly use the logic from the call method for serving
            return {'output': self(inputs, sublayer_output)}        
        return serve_dynamic
       

    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="inputs"),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="sublayer_output")])
    def serve(self, inputs, sublayer_output):
        # Directly use the logic from the call method for serving
        return {'output': self(inputs, sublayer_output)}


