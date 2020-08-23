import os

def file_renamer():
    # find the directory where the files are located using the relative location of the current script
    directory = '../data/Local_Data/Images'
    # Change the current python directory to where the files are located
    os.chdir(directory)
    # Get absolute path to the files
    directory = os.getcwd()
    # Get all the filenames
    filenames = os.listdir(directory)
    # sorted(mylist, key=WhatToSortBy)
    # https://stackoverflow.com/questions/8966538/syntax-behind-sortedkey-lambda
    filenames.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    # open the notepad file
    f = open('../../train.txt', 'r')

    for file in filenames:
        # read the speed in each line and append it to the filename after an '_'
        speed = f.readline()
        new_name = file.rsplit('.')[0] + '_' + speed.rsplit('\n')[0] + '.jpg'
        os.rename(file, new_name)

    f.close()
    # Change the current python directory to where this script file is
    os.chdir(os.path.dirname(os.path.realpath(__file__)))