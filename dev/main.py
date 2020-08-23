from keras_models import *
from helper_functions import *


if __name__ == '__main__':
    #model = VGG_model(150,150)
    #model.fit(x_train, y_train)

    # 1-100
    # 75-175
    # 150-250
    # 225-325
    #
    # {20400,480,640,3}

    extract_frames = False
    save_images_as_numpy_array = True
    image_directory = '../data/Images'
    array_file = '../../data/np_array_of_images.npz'
    if extract_frames:
        frame_extractor()
        # renaming of files is done after frames are extracted
        file_renamer()
    if save_images_as_numpy_array:
        save_images_as_np_array(image_directory=image_directory, array_filename=array_file)
        array_of_images = load_numpy_array(array_filename=array_file)


    print('Done')