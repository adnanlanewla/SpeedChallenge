import tensorflow as tf

def VGG_model(image_w, image_h, l1=0, l2=0):
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
    #new_model.add(tf.keras.layers.Dense(5, activation='softmax', name='new_predictions'))

    # new_model.summary()
    new_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    print(new_model.summary())
    return new_model


