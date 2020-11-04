import models.FlowNet_S as flownet_s
from pipeline import *
from helper_functions import  *
from imageio import imread, imwrite

#The flow output pattern matches the Pytorch output exactly. The background color of the output will depend on
#which image is the first in image concantenation.

image1 = cv2.imread(
    'C:/Users/hlane/Documents/Machine Learning/SpeedChallenge/SpeedChallenge_Local/flownet1-pytorch-master/images/flow/3d/image1.png')
image1 = cv2.imread(
    '../data/image1.png')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1 = apply_transform(image1)
image2 = cv2.imread(
    'C:/Users/hlane/Documents/Machine Learning/SpeedChallenge/SpeedChallenge_Local/flownet1-pytorch-master/images/flow/3d/image2.png')
image2 = cv2.imread(
    '../data/image2.png')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2 = apply_transform(image2)
# image = np.concatenate([image1, image2], axis=-1) for blue output background
image = np.concatenate([image2, image1], axis=-1)  # for yellow output background
image = np.reshape(image, (1, 384, 512, 6))
image = image.astype('float32')

model = flownet_s.FlowNet_S()
model.build(input_shape=(1, 384, 512, 6))
model.load_weights('../data/Local_Data/FlowNetS_Checkpoints/flownet-S2')
model.compile()
model.summary()
a = model(image)
print(a)
rgb_flow = flow2rgb(20 * a, max_value=10)
to_save = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
imwrite('../data/test' + '.png', to_save)