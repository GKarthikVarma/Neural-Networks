# Convolutional Neural-Network for MNIST #
---------------------------------------
An implementation of convolutional neural-network (CNN) for MNIST.

## Network architecture ##

CNN with 4 layers has following architecture.

+ input layer : 784 nodes (MNIST images size)
+ first convolution layer : 5x5x32
+ first max-pooling layer
+ second convolution layer : 5x5x64
+ second max-pooling layer
+ third fully-connected layer : 1024 nodes
+ output layer : 10 nodes (number of class for MNIST)

## Usage
### Train
`python train.py`

Trained model is saved as "model/model.ckpt".
