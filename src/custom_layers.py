import tensorflow as tf

class FasterCosSimConv2D(tf.keras.layers.Layer):
    def __init__(self,  filters, kernel_size, strides, padding='VALID', dilation_rate=1, activation=None, use_bias=True, 
                 kernel_initializer='glorot_uniform', bias_initializer='zeros'):        
        super(FasterCosSimConv2D, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
                 
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(self.filters, 
                                           self.kernel_size, 
                                           strides=self.strides, 
                                           padding=self.padding,
                                           dilation_rate=self.dilation_rate, 
                                           activation=self.activation, 
                                           use_bias=self.use_bias,
                                           kernel_initializer=self.kernel_initializer,
                                           bias_initializer=self.bias_initializer)
        
        self.p = self.add_weight(
            shape=(self.filters,), initializer='zeros', trainable=True)

        self.q = self.add_weight(
            shape=(1,), initializer='zeros', trainable=True)


    def call(self, inputs, training=None):
        """ tf.images.extract_patches grabs patches from an input as if a sliding window
            has been passed over the input. Given that the numerator term in the sharpened
            cosine similarity formula is essentially a convolution, it is more efficient in
            Tensorflow to rely on the default convolution operator for the numerator. This 
            leaves the problem of calculating the L2-norm for the image patches. However, calling
            extract_patches against the input with the same arguments as the Conv2D layer gives us
            the necessary image patches."""
        x_patches = tf.image.extract_patches(images=inputs,
                                       sizes=[1, self.kernel_size, self.kernel_size, 1],
                                       strides=[1, self.strides, self.strides, 1],
                                       rates=[1, self.dilation_rate, self.dilation_rate, 1],
                                       padding=self.padding)
        
        
        x_feats = self.conv(inputs)
        q = tf.square(self.q) / 10
        x_norm = tf.sqrt(tf.maximum(tf.math.reduce_sum(tf.square(x_patches), axis=-1), 1e-12))[:,:,:,None] + q
        w_norm = tf.sqrt(tf.maximum(tf.math.reduce_sum(tf.square(self.conv.weights[0]),  
                                                       axis=(0,1,2)), 1e-12))[None,None,None,:] + q        
        x = x_feats/tf.multiply(x_norm, w_norm)
        x = tf.abs(x) + 1e-12
        
        sign = tf.sign(x_feats)
        x = tf.pow(x, tf.nn.softmax(self.p))
        x = sign * x
        return x


class MaxAbsPool2D(tf.keras.layers.Layer):
    "Inefficient and lazy version of MaxAbsPool2D which pools twice, but it's much shorter to write."""
    def __init__(self, pool_size):
        super(MaxAbsPool2D, self).__init__()
        self.pooling = tf.keras.layers.MaxPool2D(pool_size=pool_size)
        
    def call(self, inputs, training=None):
        pos_pool = self.pooling(inputs)
        neg_pool = self.pooling(-1*inputs)
        
        mask = tf.cast(neg_pool > pos_pool, dtype=tf.float32)
        return -1*mask*neg_pool + (1-mask)*pos_pool
