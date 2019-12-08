To run this neural network, run Main.java

The program will create a neural network with the specifications contained in the configuration file. It will then 
randomize the weights of the network and train it using the data in the training data file. The output of the network 
will be put into a bmp file specified by the user. You can override the default file paths if you wish. Your files
should have the following structures:

## Structure of the Weights File

The format for the weights is as follows: first, the size of each layer is given. Then, for each 
layer, the matrix for the weights is given. This matrix is such that the number of rows is the 
number of nodes in the previous activation layer, and the number of columns is the number of nodes 
in the next activation layer. Each column is space separated and each row is on the next line. There 
is a blank line between each layer and this format is repeated for each layer.

An example of a weights file would be:
     
    2 2 1
    
    0.5 0.5
    0.5 0.5
    
    0.3
    0.3
   
Here, there are two connectivity layers, shown by the two matrices. The first layer connects
two nodes to two nodes, and the second layer connects two nodes to one node.

## Structure of the Image Training Data File

The image training data file contains the filenames of all input and expected bmp files. These files will then be 
converted to an array and passed through the network. The structure of the file is as follows: The first line has the 
number of training cases, the height of each image and the width of each image. Then, the next lines contain the input 
file and the expected output file.

An example of a image training file is:

    2 100 100
    input1.bmp
    expected1.bmp
    input2.bmp
    expected2.bmp
    
Here, there are two training cases, with input and expected images that are 100x100 in size.

## Structure of the Training Data File

The format of the training data is as follows: On the first line, the number of test cases is given. Then, on the 
following lines, for each test case, first the input values are given, space-separated, then the expected output values 
are given space-separated.

An example of a training data file is:

    4 2 1
    0 0
    0
    0 1
    1
    1 0
    1
    1 1
    0
    
In this case, there are 4 test cases, each with 2 input nodes and 1 output node.

## Structure of the Configuration File

The format of the configuration file is as follows:
                                        
First, the size of each layer is given. Each of these sizes are given space-separated.

Each of the next lines contain a variable that configures a part of the training:

    Min Weight:         the smallest value that the weights can be randomized to
    Max Weight:         the largest value that the weights can be randomized to
    Learning Rate:      the initial learning rate of the network
    Lambda Multiplier:  how much to multiply the learning rate by each epoch
    Maximum Epochs:     the number of epochs that will be run when training the network
    Maximum Iterations: the number of times the network is retrained
    Error Threshold:    the neural net stops when it goes below this error
    Printing Rate:      how often to print the error during training