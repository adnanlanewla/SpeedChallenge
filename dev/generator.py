import tensorflow as tf
import cv2
import numpy as np
import helper_functions
from sklearn.model_selection import train_test_split

def generator_pipeline(image_directory):
    filenames, labels = helper_functions.get_filenames_labels(image_directory)
    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
        filenames, labels, test_size=0.2, random_state=1)
    batch_size = 32
    my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size)
    my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size)
    return my_training_batch_generator, my_validation_batch_generator


class My_Custom_Generator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    # This methods gets the size of the batch for X and Y
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        #TODO: we are dividing each image by 255.0, if we need to get a image of 3 channel,
        # then i am not sure if we need to do that
        return np.array([
            cv2.imread(str(file_name))
            for file_name in batch_x]) / 255.0, np.array(batch_y)