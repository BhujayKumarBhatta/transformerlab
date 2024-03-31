import tensorflow as tf
tf.random.set_seed(42)




class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)  # Ensure 'step' is a float
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        # Ensure 'd_model' is also a float for consistency
        d_model = tf.cast(self.d_model, tf.float32)
        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)


class MyCustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=400, decay_rate=0.00023):
        super(MyCustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_rate = decay_rate
        self.initial_lr = 0.0015  # Starting learning rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Exponential warm-up
        exp_warmup = tf.math.exp((step - self.warmup_steps) / self.warmup_steps)
        # Exponential decay
        exp_decay = tf.math.exp(-self.decay_rate * (step - self.warmup_steps))
        # Choose learning rate based on whether we're in warm-up or decay phase
        lr = tf.where(step <= self.warmup_steps, 
                      self.initial_lr * exp_warmup, 
                      self.initial_lr * exp_decay)
        return lr

