import os
import numpy as np


def process_data_for_convLSTM(filenames, labels, time_steps=60):
    '''
    This method makes a list of a list for filenames and labels associated with it
    each row in a list has 60 items (60 is default. Which is the number of time steps) which is equivalent to 3 seconds of video
    The inner length of the list is 60 and the outer length is 340. 20400 images / 60 = 340
    :param filenames:
    :param labels:
    :return:
    '''
    X = [[filenames[j] for j in range((i*time_steps),((i+1)*time_steps))] for i in range(len(filenames)//time_steps)]
    # map the label to float from string
    labels = list(map(float, labels))
    # finding the maximum of a list
    max_y = max(labels)
    # This line also normalizes every value in the list between 0 and 1
    Y = [[labels[j]/max_y for j in range((i*time_steps),((i+1)*time_steps))] for i in range(len(filenames)//time_steps)]
    return X, Y

if __name__ == '__main__':
    None