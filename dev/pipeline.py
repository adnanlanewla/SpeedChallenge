import cv2
from sklearn.model_selection import train_test_split
from models.keras_models import *
import os
from models.FlowNet_S import *
from generator import *
import helper_functions
import datapreprocesor
from imageio import imread, imwrite

def get_image(filename):
    '''
    This methods return an transformed image given a filename
    :param filename:
    :return:
    '''
    image1 = cv2.imread(
        filename)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = helper_functions.apply_transform(image1)
    return image1

def combine_images_for_opti_flow(image1, image2):
    '''
    This method combines the two images for optiflow
    :param image1:
    :param image2:
    :return:
    '''
    image = np.concatenate([image1, image2], axis=-1) #for yellow output background
    image = np.reshape(image, (1, 480,640,6))
    image = image.astype('float32')
    return image

def save_image(predicted_array, filename):
    '''
    The method saves the image array in to a file.
    :param predicted_array:
    :param filename: The filename should contain the full path
    :return:
    '''
    rgb_flow = helper_functions.flow2rgb(20 * predicted_array, max_value=10)
    to_save = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
    imwrite(filename, to_save)

def build_flowNetmodel(input_shape=(None,480,640,6)):
    '''
    This methods initializes the flow net model
    :param input_shape:
    :return:
    '''
    flownet_model = FlowNet_S()
    flownet_model.build(input_shape=input_shape)
    flownet_model.load_weights('../data/Local_Data/FlowNetS_Checkpoints/flownet-S2')
    flownet_model.compile()
    flownet_model.summary()
    return flownet_model

def optical_flow_pipeline(image_directory):
    '''
    This the pipeline for optical flow
    :param image_directory:
    :return:
    '''
    # This methods get the new filesname for flow net model
    filenames, accelerations,new_opti_flow_filenames =\
        helper_functions.fileIO_for_opti_flow(image_directory)
    height, width, channels = cv2.imread(os.path.join(image_directory,filenames[0])).shape
    # build the flownet model
    flownet_model = build_flowNetmodel(input_shape=(1,height,width,6))

    # Iterate over each filename and predict optical flow for it
    for i in range(len(filenames)-1):
        file1 = os.path.join(image_directory,filenames[i])
        file2 = os.path.join(image_directory,filenames[i+1])
        image1 = get_image(filename=file1)
        image2 = get_image(filename=file2)
        image = combine_images_for_opti_flow(image1, image2)
        #  Predict the optical flow the two images
        prediction_flownet = flownet_model(image)
        # Path to save the predicted image
        path = '../data/predicted_images'
        # If path doesn't exist create the path
        if not os.path.exists(path):
            os.makedirs(path)
        new_file = os.path.join(path,new_opti_flow_filenames[i])
        # save the predicted image from Optical flow
        save_image(prediction_flownet, new_file)

def VGG16_pipeline(image_directory, batch_size=32, normalize_y=False):
    '''
    Pipeline for VGG 16 model
    :param image_directory:
    :param batch_size:
    :param normalize_y:
    :return:
    '''
    # Create a custom generator
    my_training_batch_generator, my_validation_batch_generator = generator_pipeline(image_directory, batch_size, normalize_y)
    height, width, channels = cv2.imread(my_training_batch_generator.image_filenames[0]).shape
    input_shape = (height, width, channels)
    # Create a VGG 16 model
    vgg_16_model = VGG_model_function(input_shape=input_shape)
    # Fit the model
    vgg_16_model.fit_generator(my_training_batch_generator,validation_data=my_validation_batch_generator, epochs=2, use_multiprocessing=True, workers=0)
    print(vgg_16_model.reset_metrics())

def ConvLSTM_pipeline(image_directory, batch_size=32, time_steps=60):
    # Get a list of filename and labels associated with that filename
    filenames, labels = helper_functions.get_filenames_labels(image_directory)
    # Convert the filename and labels in to List of a list. The reason for this conversion is because the ConvLSTM layer in keras require a shape of
    # (seq length, rows,cols, channels)
    X, Y = datapreprocesor.process_data_for_convLSTM(filenames,labels, time_steps=time_steps)

    # split in to Train and Test set.
    # TODO: shuffle option is set to false. We should also try when then option is set to True
    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=1, shuffle=False)

    # Custom training and validation generator for ConvLSTM model
    my_training_batch_generator = my_convLSTM_generator(X_train_filenames, y_train, batch_size)
    my_validation_batch_generator = my_convLSTM_generator(X_val_filenames, y_val, batch_size)
    # defining the input shape
    input_shape = (my_training_batch_generator.length_of_one_sample,
                   my_training_batch_generator.height, my_training_batch_generator.width, my_training_batch_generator.channels)
    # Create a ConvLSTM model. The default filter value is 64 and the default value for dense unit is 256.
    # I have changed these values to not have too many trainable parameteres
    model = Conv_LSTM_function(input_shape=input_shape, dense_units=8, filters=16)
    # define a early stopping mechanism incase the accuracy doesn't improve in few epochs
    earlystop = tf.keras.callbacks.EarlyStopping(patience=7)
    callbacks = [earlystop]
    # Calling the fit method
    model.fit_generator(my_training_batch_generator,epochs=2,validation_data=my_validation_batch_generator, callbacks=callbacks,shuffle=False)
    print(model.metrics)

