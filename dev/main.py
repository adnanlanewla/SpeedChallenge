from models.keras_models import *
from helper_functions import *
from image_preprocessing import *
import models.FlowNet_S as flownet_s
from optical_flow_pipeline import *
from helper_functions import  *
from imageio import imread, imwrite
import generator

#The flow output pattern matches the Pytorch output exactly. The background color of the output will depend on
#which image is the first in image concantenation.
if __name__ == '__main__':
    image1 = cv2.imread(
        'C:/Users/hlane/Documents/Machine Learning/SpeedChallenge/SpeedChallenge_Local/flownet1-pytorch-master/images/flow/3d/image1.png')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = apply_transform(image1)
    image2 = cv2.imread(
        'C:/Users/hlane/Documents/Machine Learning/SpeedChallenge/SpeedChallenge_Local/flownet1-pytorch-master/images/flow/3d/image2.png')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2 = apply_transform(image2)
    # image = np.concatenate([image1, image2], axis=-1) for blue output background
    image = np.concatenate([image2, image1], axis=-1) #for yellow output background
    image = np.reshape(image, (1, 384, 512, 6))
    image = image.astype('float32')

    model = flownet_s.FlowNet_S()
    model.build(input_shape=(None, 384, 512, 6))
    model.load_weights('../data/Local_Data/FlowNetS_Checkpoints/flownet-S2')
    model.compile()
    model.summary()
    a = model(image)
    print(a)
    rgb_flow = flow2rgb(20 * a, max_value=10)
    to_save = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
    imwrite('../data/test' + '.png', to_save)

    # extract_frames = False
    # generate_pipeline = True
    # image_directory = "..\data\Images"
    # if extract_frames:
    #     # renaming of files is done during extraction of frames
    #     frame_extractor()
    #
    # # this will generate the pipeline, we can use for training the keras model
    # if generate_pipeline:
    #     my_training_batch_generator, my_validation_batch_generator = generator.My_Custom_Generator(image_directory)
    #
    # optical_flow_pipeline_with_VGG_16(image_directory)
    # print('Done')
