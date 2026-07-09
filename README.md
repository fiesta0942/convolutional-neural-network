Convolutional Neural Network
Based on Deep Learning Specialization, deeplearning.ai, this repository keeps track of my trial on ResNet

1. ResNet 18
- Using PyTorch, I've implemented basic Residual Block with two 3x3 convolution layer with ReLU activation function, a shortcut path and 1x1 convolution to match the dimension of images
- With that, I've made ResNet-18 with 4 residual layers, each stacked with two Res Block, average pooling layer and fully connected layer in the end
- For better performance, I've also implemented mini - batch gradient descent with Adam Optimization. (Batch Normalization is NOT yet implemented)

2. Simple Object Detection with YOLO