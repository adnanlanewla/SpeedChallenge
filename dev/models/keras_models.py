import tensorflow as tf

def VGG_model_function(image_w, image_h, l1=0, l2=0):
    model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(image_w, image_h, 3))
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

class VGG_model(tf.keras.Model):

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.VGG16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

        for layer in self.VGG16.layers[0:19]:
            layer.trainable = False

        self.regularizer = tf.keras.regularizers.l1()
        self.sequential= tf.keras.models.Sequential()
        self.flatten_layer = tf.keras.layers.Flatten(name='flatten')
        self.dense1 = tf.keras.layers.Dense(256, activation='softmax', name='new_fc1', kernel_regularizer=self.regularizer)
        self.dense2 = tf.keras.layers.Dense(1, activation='linear', name='new_predictions')


    def call(self, inputs, training=None, mask=None):

        VGG16_models = self.VGG16(inputs)
        print(VGG16_models.summary())
        model = self.sequential(VGG16_models)
        model = self.flatten_layer(model)
        model = self.dense1(model)
        model = self.dense2(model)
        print(model.summary())
        return model

def linear_reg_keras(input_shape):
    model = tf.keras.Sequential([
        tf.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.layers.Dense(64, activation='relu'),
        tf.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model