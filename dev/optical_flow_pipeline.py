import cv2
from sklearn.model_selection import train_test_split
from models.keras_models import *
import os
from models.FlowNet_S import *
from generator import *
import helper_functions
from imageio import imread, imwrite

def get_image(filename):
    image1 = cv2.imread(
        filename)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = helper_functions.apply_transform(image1)
    return image1

def combine_images_for_opti_flow(image1, image2):
    image = np.concatenate([image1, image2], axis=-1) #for yellow output background
    image = np.reshape(image, (1, 480,640,6))
    image = image.astype('float32')
    return image

def save_image(predicted_array, filename):
    rgb_flow = helper_functions.flow2rgb(20 * predicted_array, max_value=10)
    to_save = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
    imwrite(filename, to_save)

def build_flowNetmodel(input_shape=(None,480,640,6)):
    flownet_model = FlowNet_S()
    flownet_model.build(input_shape=input_shape)
    flownet_model.load_weights('../data/Local_Data/FlowNetS_Checkpoints/flownet-S2')
    flownet_model.compile()
    flownet_model.summary()
    return flownet_model

def optical_flow_pipeline_with_VGG_16(image_directory):
    flownet_model = build_flowNetmodel(input_shape=(1,480,640,6))
    filenames, accelerations,new_opti_flow_filenames =\
        helper_functions.fileIO_for_opti_flow(image_directory)

    for i in range(len(filenames)-1):
        file1 = os.path.join(image_directory,filenames[i])
        file2 = os.path.join(image_directory,filenames[i+1])
        image1 = get_image(filename=file1)
        image2 = get_image(filename=file2)
        image = combine_images_for_opti_flow(image1, image2)
        prediction_flownet = flownet_model(image)
        path = '../data/predicted_images'
        if not os.path.exists(path):
            os.makedirs(path)
        new_file = os.path.join(path,new_opti_flow_filenames[i])
        save_image(prediction_flownet, new_file)

    # my_training_batch_generator, my_validation_batch_generator = generator_pipeline(path)
    # vgg_model = VGG_model((480,640,3))
    # vgg_model.compile(loss="mse", optimizer='adam', metrics=["mse", 'mae'])
    # vgg_model.fit_generator(my_training_batch_generator,validation_data=my_validation_batch_generator)

def optical_flow_pipeline_with_linear_regression(image_directory):
    flownet_model = FlowNet_S()
    filenames, labels = helper_functions.fileIO_for_opti_flow(image_directory)

    for i in range(len(filenames)):
        file = filenames[i]
        label = labels[i]
        image_array = cv2.imread(file)
        prediction_flownet = flownet_model.predict(image_array)
        path, filename = os.path.split(file)
        path = os.path.join(path,"flownet_predicted")
        if not os.path.exists(path):
            os.makedirs(path)
        new_file = os.path.join(path,filename)
        cv2.imwrite(new_file, prediction_flownet)

    my_training_batch_generator, my_validation_batch_generator = generator_pipeline(path)

    linear_reg_model = linear_reg_keras((480,640,3))
    linear_reg_model.fit_generator(my_training_batch_generator,validation_data=my_validation_batch_generator)