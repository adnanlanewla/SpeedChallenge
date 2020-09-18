import cv2
from sklearn.model_selection import train_test_split
from models.keras_models import *
import os
from models.FlowNet_S import *
from generator import *


def optical_flow_pipeline_with_VGG_16(image_directory):
    flownet_model = FlowNet_S()
    filenames, labels = helper_functions.get_filenames_labels(image_directory)

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
    vgg_model = VGG_model((480,640,3))
    vgg_model.compile(loss="mse", optimizer='adam', metrics=["mse", 'mae'])
    vgg_model.fit_generator(my_training_batch_generator,validation_data=my_validation_batch_generator)

def optical_flow_pipeline_with_linear_regression(image_directory):
    flownet_model = FlowNet_S()
    filenames, labels = helper_functions.get_filenames_labels(image_directory)

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