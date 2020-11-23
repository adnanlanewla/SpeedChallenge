# SpeedChallenge
Goal is to predict the speed of a car from a video

# Challenge Description
Description by [comma.ai](comma.ai)

  Welcome to the comma.ai 2017 Programming Challenge!

  Basically, your goal is to predict the speed of a car from a video.

  data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
  data/train.txt contains the speed of the car at each frame, one speed on each line.

  data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.
  Your deliverable is test.txt

  We will evaluate your test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.

# Data Processing
The original video data was converted in to images files using cv2. Each image file was named by following this nomenclature directory_frame_minute_second_frame_number_within_one_second_speed.jpg

The method used for extracting images also has an option if the user wants to reduce the size of the original images by settings the rescale_factor

# Model Exploration
Three methods were used to train and evaluate our data.

1) [Flownet 1.0](https://arxiv.org/abs/1504.06852) - 
we used existing implementation of Flownet1.0 in pytorch and converted that in to keras. Here's the implementation of Flownet 1.0 in pytorch and the corresponding weights https://github.com/sampepose/flownet2-tf

2) VGG 16 - 
VGG-16 Convolution neural netowrk was used to process individual images with its label.

3) ConvLSTM2D - 
  COnvolutions LSTM model was used where the input transformations and recurrent transformations are both convolutional


# Sources
[1] A. Dosovitskiy, P. Fischer, E. Ilg, P. Häusser, C. Hazırba¸s, V. Golkov, P. v.d. Smagt, D. Cremers, and T. Brox. Flownet: Learning optical flow with convolutional networks. In IEEE International Conference on Computer Vision (ICCV), 2015.

[2] Sam Pepose, FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks, https://github.com/sampepose/flownet2-tf



