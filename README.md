# cnn-week10
CNN and MNIST using Go

### Computer Vision (MNSIT images), Neural Networks (NNs), and Convolutional Neural Networks (CNNs)  
This repo sets up some Go code to read in MNIST images leveraging the [GoMNIST](https://github.com/petar/GoMNIST) repo. It loads in all the images and displays some examples in grayscale. See example image below:  
![GrayScale](Grayscale_Example.png)  
After reading in the images, the next step was to build a model to make image predictions. Here are the steps for this project:  
1. Split the MNIST images into test and training
2. Find a Go Package to build the neural network [go-deep](https://github.com/patrikeh/go-deep)  
    a. This is not a CNN  
    b. Will start with a simple NN and can build after getting the code to work  
3. Set up the params for the neural network and create functions for training
    a. Normalize the images  
    b. Convert labels and build examples  
    c. One hot encoding  
4. Run the code - go run main.go or double click the app - mnistNN.app  

### *Results*  


### Future Steps  
To continue to build on this neural network one would need a new package, perhaps [Gorgonia](https://gorgonia.org/). I attempted to use this on the MNIST images, but was unable to load the data even following this [example](https://gorgonia.org/tutorials/mnist/).  The example does a great job of showing how to build a simple CNN, but it did not integrate well with the previous work around MNIST for [isolation forests](https://github.com/ddecoen/mnist_iforest).