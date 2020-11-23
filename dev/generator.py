import tensorflow as tf
import cv2
import numpy as np
import helper_functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def generator_pipeline(image_directory, batch_size=32, normalize_y=False):
    '''
    This is generator pipeline.
    :param image_directory:
    :param batch_size:
    :param normalize_y:
    :return:
    '''
    # Get a list of filenames and labels from a directory
    filenames, labels = helper_functions.get_filenames_labels(image_directory)
    # Split the data in to train and test set
    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
        filenames, labels, test_size=0.2, random_state=1)
    # Create a custom generator for training data
    my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size, normalize_y)
    # Create a custom generator for validation data
    my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size, normalize_y)

    # return both the generators
    return my_training_batch_generator, my_validation_batch_generator

# My custom generator class
class My_Custom_Generator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, normalize_y=False):
        self.image_filenames = image_filenames
        self.labels = [float(i) for i in labels]
        self.batch_size = batch_size
        self.normalize_y = normalize_y
        if self.normalize_y:
            scaler = MinMaxScaler()
            data = np.array(labels)
            data = data.reshape(-1,1)
            scaler.fit(data)
            transformed_data = scaler.transform(data)
            transformed_data = transformed_data.reshape(-1)
            transformed_data = transformed_data.tolist()
            data = data.reshape(-1)
            data = data.tolist()
            if data == labels:
                self.labels = transformed_data
            else:
                raise Exception('List Not equal')

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    # This methods gets the size of the batch for X and Y
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]


        #TODO: we are dividing each image by 255.0, if we need to get a image of 3 channel,
        # then i am not sure if we need to do that
        return np.array([
            cv2.imread(file_name)
            for file_name in batch_x]) / 255.0, np.array(batch_y)


# my custom generator class for convLSTM
class my_convLSTM_generator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.height, self.width, self.channels = cv2.imread(image_filenames[0][0]).shape
        self.length_of_one_sample = len(image_filenames[0])


    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    # This methods gets the size of the batch for X and Y
    def __getitem__(self, idx):

        # we get the batch list of X and Y by going from [batch_size : batch_size + 1]
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Initialize the array X and Y with correct shapes
        array_x = np.zeros((len(batch_x),self.length_of_one_sample, self.height, self.width,self.channels))
        array_y = np.zeros((len(batch_y), self.length_of_one_sample, 1))
        # iterate over the batch size and create an image array for it
        # This is outer loop. The size of the outer loop is batch size
        for index, val in enumerate(batch_x):
            # This is the for loop for inner list which has 60 images i.e images for 3 secs since each sec has 20 frames
            for inner_index, inner_value in enumerate(batch_x[index]):
                array_x[index,inner_index] = cv2.imread(inner_value)
                array_y[index, inner_index] = batch_y[index][inner_index]

        return array_x, array_y
        # return np.array([
        #     cv2.imread(file_name)
        #     for file_name in batch_x]) / 255.0, np.array(batch_y)