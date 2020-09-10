import tensorflow as tf


def conv2D(batchNorm, filters, kernel_size=3, stride=1):
    if batchNorm:
        convolution_layer = tf.keras.Sequential(
            [
                tf.keras.layers.ZeroPadding2D(padding=(kernel_size-1)//2),
                tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, use_bias=False),
                tf.keras.layers.BatchNormalization(axis=-1),
                tf.keras.layers.LeakyReLU(0.1)
            ]
        )
    else:
        convolution_layer = tf.keras.Sequential(
            [
                tf.keras.layers.ZeroPadding2D(padding=(kernel_size - 1) // 2),
                tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, use_bias=True,
                                       bias_initializer='RandomUniform'),
                tf.keras.layers.LeakyReLU(0.1)
            ]
        )
    return convolution_layer


'''
In Transposed convolution2D in Pytorch, the padding argument effectively adds dilation * (kernel_size - 1) - padding 
amount of zero padding to both sizes of the input. This is set so that when a Conv2d and a ConvTranspose2d are 
initialized with same parameters, they are inverses of each other in regard to the input and output shapes. The padding
here is therefore calculated using 'kernel_size-1-padding' since dilation is (1,1).
'''


def deconv(filters, kernel_size=4, padding=1, stride=2, bias=True, activation='LeakyReLU'):
    sequential_model = tf.keras.Sequential()
    sequential_model.add(tf.keras.layers.ZeroPadding2D(kernel_size-1-padding))
    sequential_model.add(tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, stride=stride,
                                                         use_bias=bias, bias_initializer='RandomUniform'))
    if activation:
        sequential_model.add(tf.keras.layers.LeakyReLU(0.1))

    return sequential_model


def predict_flow():
    return tf.keras.Sequential(
        [
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(filters=2, kernel_size=3, stride=1, bias=True, bias_initializer='RandomUniform')
        ]
    )