import os
import numpy as np

def process_data_for_convLSTM(filenames, labels):
    X = [[filenames[j] for j in range((i*60),((i+1)*60))] for i in range(len(filenames)//60)]
    labels = list(map(float, labels))
    max_y = max(labels)
    Y = [[labels[j]/max_y for j in range((i*60),((i+1)*60))] for i in range(len(filenames)//60)]
    return X, Y

if __name__ == '__main__':
    None