import tensorflow as tf
import cv2
import numpy as np
import helper_functions
from sklearn.model_selection import train_test_split
from generator import *


def optical_flow_pipeline(image_directory):
    filenames, labels = helper_functions.get_filenames_labels(image_directory)
    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
        filenames, labels, test_size=0.2, random_state=1)
    batch_size = 32
    my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size)
    my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size)
    return my_training_batch_generator, my_validation_batch_generator