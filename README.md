# a multi-scale 3D convolutional neural network is used in hyperspectral image classification. 
## Overview: 
this is my graduation project that classify indian_pines、Pavia_university and Salinas dataset based on deep_learning.
[download the dataset](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
## Language:
Python3.x
## Tools:
Tensorflow
## Describe：
The sensor source of image does not require much pre-processing, in order to reduce the burden on the computer, the input data is compressed using PCA and 96% effective spectral information is extracted. The data is normalized and input into a three-dimensional convolutional network to get feature of empty and spectrum. The three-dimensional convolutional network uses multi-scale convolution kernels to extract multi-scale spatial features, then the joint feature maps of the spectral and spatial properties of the hyperspectral image fed through a fully connected layer, which finally predicts each pixel label through the Softmax classifier.
