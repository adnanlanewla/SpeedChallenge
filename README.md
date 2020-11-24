# SpeedChallenge
Goal is to predict the speed of a car from a video

# Challenge Description
Description by [comma.ai](comma.ai)

  Welcome to the comma.ai 2017 Programming Challenge!

  Basically, the goal is to predict the speed of a car from a video.

  data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
  data/train.txt contains the speed of the car at each frame, one speed per frame.

  data/test.mp4 is a different driving video containing 10798 frames. Video is also shot at 20 fps.
  The deliverable is test.txt

  We will evaluate the test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.

# Data Processing
The original video data was converted in to images files using cv2. Each image file was named by following this nomenclature directory_frame_minute_second_frame_number_within_one_second_speed.jpg

The method used for extracting images also has an option to reduce the size of the original images by setting the rescale_factor

# Model Exploration
The below methods were used to train and evaluate our data.

1) [Flownet 1.0](https://arxiv.org/abs/1504.06852)
* We used the existing implementation of Flownet1.0 in pytorch and converted that in to tensorflow 2.x. Below is the link for the implementation of Flownet 1.0 in pytorch and the corresponding weights (https://github.com/hlanewala/FlowNet_1.0).
We used the flownet model in conjunction with:
  -  ### Linear Regression:
      The output of the optical flow model is flattened and fed in to the linear regression model, using ordinary least squares with regularization as the loss function. The model did not give satisfactory results, with an R<sup>2</sup> value of 0.3 on the test set. There are likely two main reasons for this: a) we did not retrain the flownet model to fine-tune the parameters of the original optical flow model due to limited computational resources and b) the FlowNet1.0 model, by Dosovitskiy et al, was trained on images where the motion between the two consecutive frames was simulated using planar transformations. In our video footage, the changes between the two consecutive frames is mostly a change in the depth perception, which is not handled well by this model. There is a newer version of the optical flow model, FlowNet2.0, which is supposed to handle the change in depth much more accurately but application of this model on the CPU is not straightforward and we expect to explore that in the future. 

  - ### VGG-16
    The idea behind using the flownet model for this problem is to learn the optical flow between two consecutive images in the video and using that optical flow representation to estimate the acceleration (difference between speeds of two consecutive frames). The output dense unit was used to predict the acceleration using the loss function of mean squared error. The accuracy for this model was very low, this was excepted since the predicted pictures from the flownet model didn't give a clear idea on the movement between two frames. This is expected since as explained above flownet 1.0 performs poorly with a change in the depth perception.

2) [ConvLSTM2D](https://keras.io/api/layers/recurrent_layers/conv_lstm2d/)
* Convolutions LSTM model is used where the input transformations and recurrent transformations are both convolutional. This model is used widely in data involving temporal - spatial features. The idea behind the Convolutional LSTM layer is that it's a layer of LSTM units but the input and output are 3D. 
The results from this model was not satisfactory, one of the reasons is that we only used 3 second timeframe to predict the speed of the car. This 3 second window is not enough to fully train the model. We could have increased our time dimension to include a 10 second clip of our video, but that was hard to train due to the computational resource required to train that big of a network. Further research is required for this model. We hope to solve this problem by some image augmentation to reduce the features space in our images and by training it on GPU which can speed up the process for training. 


# Sources
[1] A. Dosovitskiy, P. Fischer, E. Ilg, P. Häusser, C. Hazırba¸s, V. Golkov, P. v.d. Smagt, D. Cremers, and T. Brox. Flownet: Learning optical flow with convolutional networks. In IEEE International Conference on Computer Vision (ICCV), 2015.

[2] Sam Pepose, FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks, https://github.com/sampepose/flownet2-tf



