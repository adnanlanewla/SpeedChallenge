import tensorflow as tf
from .submodules import *

class FlowNet_S(tf.keras.Model):
     def __init__(self, batchNorm=True):
         super().__init__()
         self.batchNorm = batchNorm
         self.conv1 = conv2D(self.batchNorm, filters=64, kernel_size=7, stride=2)
         self.conv2 = conv2D(self.batchNorm, filters=128, kernel_size=5, stride=2)
         self.conv3 = conv2D(self.batchNorm, filters=256, kernel_size=5, stride=2)
         self.conv3_1 = conv2D(self.batchNorm, filters=256)
         self.conv4 = conv2D(self.batchNorm, filters=512, stride=2)
         self.conv4_1 = conv2D(self.batchNorm, filters=512)
         self.conv5 = conv2D(self.batchNorm, filters=512, stride=2)
         self.conv5_1 = conv2D(self.batchNorm, filters=512)
         self.conv6 = conv2D(self.batchNorm, filters=1024, stride=2)
         self.conv6_1 = conv2D(self.batchNorm, filters=1024)

         self.deconv5 = deconv(512)
         self.deconv4 = deconv(256)
         self.deconv3 = deconv(128)
         self.deconv2 = deconv(64)

         self.predict_flow6 = predict_flow()
         self.predict_flow5 = predict_flow()
         self.predict_flow4 = predict_flow()
         self.predict_flow3 = predict_flow()
         self.predict_flow2 = predict_flow()

         self.upsampled_flow6_to_5 = deconv(2, bias=False, activation=False)
         self.upsampled_flow5_to_4 = deconv(2, bias=False, activation=False)
         self.upsampled_flow4_to_3 = deconv(2, bias=False, activation=False)
         self.upsampled_flow3_to_2 = deconv(2, bias=False, activation=False)

         self.upsample1 = tf.keras.layers.UpSampling2D(size=(4,4))

     def call(self, inputs, training=None):
         return None
