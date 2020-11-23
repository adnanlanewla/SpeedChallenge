import tensorflow as tf


def Conv_LSTM_function_simplified(input_shape=(60,480,640,3), filter1=4, filter2=8,filter3=16, dense_units=128, output_units=60):
    '''
    Creates a ConvLSTM model with multiple ConvLSTM2D layers. These multiple conv LSTM are neccasary to reduce the parameter training for the overall model.
    :param input_shape: Input shape should be (seq_len, img_height, img_width, 3). THe batch size should not be part of input_shape
    :return:
    '''
    model = tf.keras.Sequential()
    # Creates a ConVLSTM2D layer
    model.add(tf.keras.layers.ConvLSTM2D(filters=filter1, kernel_size=(7, 7), return_sequences=True, data_format="channels_last",input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1,3,3), data_format="channels_last",padding='valid'))
    model.add(tf.keras.layers.ConvLSTM2D(filters=filter2, kernel_size=(5, 5), return_sequences=True, data_format="channels_last"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1,3,3), data_format="channels_last",padding='valid'))
    model.add(tf.keras.layers.ConvLSTM2D(filters=filter3, kernel_size=(3, 3), return_sequences=True, data_format="channels_last"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1,3,3), data_format="channels_last",padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense_units, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(output_units, activation='softmax'))
    print(model.output_shape)
    print(model.summary())
    model.compile(loss='mse',optimizer='adam', metrics=["accuracy"])
    return model

def Conv_LSTM_function(input_shape=(60,480,640,3), filters=64, dense_units=256, output_units=60):
    '''
    Creates a ConvLSTM model.
    :param input_shape: Input shape should be (seq_len, img_height, img_width, 3). THe batch size should not be part of input_shape
    :return:
    '''
    model = tf.keras.Sequential()
    # Creates a ConVLSTM2D layer
    model.add(tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=(3, 3), return_sequences=True, data_format="channels_last",
                         input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense_units, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(output_units, activation='softmax'))
    print(model.output_shape)
    print(model.summary())
    model.compile(loss='mse',optimizer='adam', metrics=["accuracy"])
    return model

def VGG_model_function(input_shape=(120,160,3), l1=0, l2=0):
    '''
    Creates a VGG 16 model with one dense unit at the end. Only the top dense layer is trainable. All the convolution layers are not trainable
    :param input_shape:
    :param l1:
    :param l2:
    :return:
    '''
    model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    print(model.summary())
    for layer in model.layers[0:19]:
        layer.trainable = False

    if l1 > 0:
        regularizer = tf.keras.regularizers.l1(l1)
    else:
        regularizer = tf.keras.regularizers.l2(l2)

    new_model = tf.keras.models.Sequential()
    new_model.add(model)
    new_model.add(tf.keras.layers.Flatten(name='flatten'))
    new_model.add(tf.keras.layers.Dense(256, activation='softmax', name='new_fc1', kernel_regularizer=regularizer))
    new_model.add(tf.keras.layers.Dense(1, activation='linear', name='new_predictions'))

    # new_model.summary()
    new_model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])
    print(new_model.summary())
    return new_model

# VGG 16 model class
class VGG_model(tf.keras.Model):

    def __init__(self, input_shape):
        super().__init__()
        #self.input_shape = input_shape
        self.VGG16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

        for layer in self.VGG16.layers[0:19]:
            layer.trainable = False

        self.regularizer = tf.keras.regularizers.l1()
        self.sequential= tf.keras.models.Sequential()
        self.flatten_layer = tf.keras.layers.Flatten(name='flatten')
        self.dense1 = tf.keras.layers.Dense(256, activation='softmax', name='new_fc1', kernel_regularizer=self.regularizer)
        self.dense2 = tf.keras.layers.Dense(1, activation='linear', name='new_predictions')


    def call(self, inputs, training=None, mask=None):
        # defining forward pass
        VGG16_models = self.VGG16(inputs)
        model = self.sequential(VGG16_models)
        model = self.flatten_layer(model)
        model = self.dense1(model)
        model = self.dense2(model)
        return model

