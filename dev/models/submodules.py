import tensorflow as tf


def conv2D(batchNorm, filters, kernel_size=3, stride=1, name=None):
    if batchNorm:
        convolution_layer = tf.keras.Sequential(
            [
                tf.keras.layers.ZeroPadding2D(padding=(kernel_size-1)//2),
                tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, use_bias=False), #Bias was false in the original model
                tf.keras.layers.BatchNormalization(axis=-1),
                tf.keras.layers.LeakyReLU(0.1)
            ], name= name
        )
    else:
        convolution_layer = tf.keras.Sequential(
            [
                tf.keras.layers.ZeroPadding2D(padding=(kernel_size - 1) // 2),
                tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, use_bias=True, #Bias was true in the original model
                                       bias_initializer='RandomUniform'),
                tf.keras.layers.LeakyReLU(0.1)
            ], name=name
        )
    return convolution_layer


'''
In Transposed convolution2D in Pytorch, the padding argument effectively adds dilation * (kernel_size - 1) - padding 
amount of zero padding to both sizes of the input. This is set so that when a Conv2d and a ConvTranspose2d are 
initialized with same parameters, they are inverses of each other in regard to the input and output shapes. The padding
here is therefore calculated using 'kernel_size-1-padding' since dilation is (1,1).
'''


def deconv(filters, kernel_size=4, stride=2, bias=False, activation='LeakyReLU', name=None):
    sequential_model = tf.keras.Sequential(name=name)
#The purpose of the zeropadding layer below is really just to prevent errors while loading the weights in to the model. This is because the model weights were originally saved with this layer and this layer became part of the computation graph. The zero padding with padding of 0 is meaningless.
    sequential_model.add(tf.keras.layers.ZeroPadding2D(0))
    sequential_model.add(tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=stride,
                                                         use_bias=bias, bias_initializer='RandomUniform')) #Bias is false in the original model
    if activation:
        sequential_model.add(tf.keras.layers.LeakyReLU(0.1))

    return sequential_model


def predict_flow(name=None):
    return tf.keras.Sequential(
        [
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=1, use_bias=False, bias_initializer='RandomUniform') #Bias is false in the original model
        ], name=name
    )

#The antipadding basically chops the output by the padding amount. The antipadding is a very important part of the
#refinement of the coarse feature maps in the optical flow network. This is equivalent to specifying the padding
# in Pytorch for the ConvTranspose2d layer.
def antipad(tensor, padding=1):
    batch, h, w, c = tensor.shape.as_list()
    # The if condition below is just a dummy condition to allow successful build of the model
    if batch == None:
        batch = 0
    # 2 below in the size argument is used for symetric 'antipadding'
    return tf.slice(tensor, begin=[0, padding, padding, 0], size=[batch, h - 2 * padding, w - 2 * padding, c])

def crop_like(inputs, target):
    if (inputs.shape[1] == target.shape[1]) and (inputs.shape[2] == target.shape[2]):
        return inputs
    else:
        return inputs[:, :target.shape[1], :target.shape[2], :]