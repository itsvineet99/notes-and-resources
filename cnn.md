## **chapter 14:** Classifying Images with Deep Convolutional Neural Networks

`page no. 451 to 498`

`ch started at 14-09-2025 10.00 am` -> `ch finished at `

### overview

- [ ] building blocks of cnn
	- [x] feature hierarchies / feature extraction 
	- [x] convolutional layers + pooling layers
	- [ ] discrete convolutional in one dimension, kernel size, stride 
	- [ ] padding 
	- [ ] size of convolutional output 
	- [ ] discrete convolution in 2D
	- [ ] naive implementation of 1D convolution
	- [ ] naive implementation of 2D convolution 
	- [ ] subsampling layers i.e max pooling and mean pooling 
- [ ] L2 regularization and dropout 
- [ ] Loss functions for classifications 
- [ ] implementing CNN in pytorch (on MNIST)
	- [ ] dealing with multiple inputs or color channels
	- [ ] CNN arch
- [ ] smile classification project using CNN
	- [ ] image transformation and data augmentation 

### Understanding CNNs and feature hierarchies

- CNN's were inspired by how visual cortex of human brain works.
- their exists layers of visual cortex, primary layers -> edges, straight lines, simple features, higher order-layers -> complex shapes and features.
- Yann LeCun and his colleagues invented CNN in 1989 and then in 2019 he received turing award.
- convolutional architecture -> feature extraction layers 
- in CNN, the early (layers that are right after input layer) extract low-level features from raw data and the later layers (fully connected mlp layers) use these features to make prediction 
- cnn extracts low-level feature and then combines them to build high level features, like in case of images they will extract features like edges, blobs, straight lines and then combine them to build to complex shapes like buildings, cats or dogs. this is how we construct so called **feature hierarchy**.

- in CNN we create a **feature map** from input image, where each element of feature map comes from local patch of pixels in the input image. this local path is also called **local receptive field**.

![feature map diagram](https://i.ibb.co/23ZCrhk8/Screenshot-2025-09-14-at-12-11-22-PM.png)


why cnn's perform better on images?

- **sparce connectivity**: as we can see only a few pixels are connected to a single feature element (unlike mlp where entire image is connected to single feature i.e layers are fully connected)
- **parameter sharing**: same weights are used for different patches of input image. which benefits us by substantially decreasing the number of weights used in the network 

- CNN -> convolutional layers + subsampling layers (pooling layers) + fully connected layer (mlp)
- pooling layers don't have biases and weights but convolutional layer and fully connected layers do.

### discrete convolutional in one dimension 