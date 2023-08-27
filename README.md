# cnn-week10
CNN and MNIST using Go

### Computer Vision (MNSIT images) and Convolutional Neural Networks (CNNs)  
This repo sets up some Go code to read in MNIST images leveraging the [GoMNIST](https://github.com/petar/GoMNIST) repo. It loads in all the images and displays some examples in grayscale. See example image below:  
![GrayScale](Grayscale_Example.png)  
After reading in the images, the next step was to build a model to make image predictions. Here are the steps for this project:  
1. Split the MNIST images into test and training
2. Find a Go Package to build the neural network [go_deep](https://github.com/patrikeh/go-deep)  
    a. This is not a CNN  
    b. Will start with a simple NN and can build after getting the code to work  
3. 