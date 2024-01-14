import tensorflow as tf


"""Helper custom layers"""


class ResidualConv2D(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(self.num_filters, self.kernel_size, padding='same', activation='relu')
        if input_shape[-1] != self.num_filters:
            self.match_channels = tf.keras.layers.Conv2D(self.num_filters, (1, 1), padding='same', activation='relu')
        else:
            self.match_channels = None

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        if self.match_channels:
            inputs = self.match_channels(inputs)
        x += inputs
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'num_filters': self.num_filters, 'kernel_size': self.kernel_size})
        return config


class UNetEncoder(tf.keras.layers.Layer):
    def __init__(self, num_filters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_block = ResidualConv2D(num_filters, kernel_size=(3, 3))
        self.downsample = tf.keras.layers.Conv2D(num_filters * 2, (3, 3), strides=(2, 2), padding='same')

    def call(self, inputs):
        residual_output = self.residual_block(inputs)
        downsampled_output = self.downsample(residual_output)
        return residual_output, downsampled_output

    def get_config(self):
        config = super().get_config()
        config.update({'num_filters': self.num_filters})
        return config



class UNetDecoder(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    def build(self, input_shape):
        super().build(input_shape)
        bs, w, h, c = input_shape[0]

    def call(self, inputs):
        return inputs


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_filters = num_filters

    def build(self, input_shape):
        # Couche de convolution transposée pour agrandir l'image
        self.transposed_conv = tf.keras.layers.Conv2DTranspose(self.num_filters, (3, 3), padding='same', activation='relu')

        super().build(input_shape)

    def call(self, inputs):
        encoder_output, skip_connection = inputs

        # Convolution transposée pour agrandir l'image
        upsampled_output = self.transposed_conv(encoder_output)

        # Concaténation avec la skip_connection
        concatenated_output = tf.keras.layers.concatenate([upsampled_output, skip_connection])

        # Opérations de convolution résiduelle
        x = ResidualConv2D(self.num_filters, kernel_size=(3, 3))(concatenated_output)
        x = ResidualConv2D(self.num_filters, kernel_size=(3, 3))(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'num_filters': self.num_filters})
        return config
