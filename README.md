To run this neural network, run MinimizeError.java

The program will create a neural network with the specifications contained in the configuration file. It will then repeatedly randomize the weights of the network and train it using the data in the training data file. If the error decreases, it will print the output of the neural network and store the weights in the weights file.

You can override the default file paths if you wish. Your files should have the following structures:

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

## Structure of the Training Data File

The format of the training data is as follows: On the first line, the number of test cases is given. Then, on the following lines, for each test case, first the input values are given, space-separated, then the expected output values are given space-separated.

An example of a training data file is:

    4
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

Lambda Multiplier - how much to multiply the learning rate by each epoch\
Learning Rate - the initial learning rate of the network\
Maximum Epochs - the maximum number of epochs that will be run when training the network\
Maximum Iterations - the maximum number of times to randomize the weights of the network and retrain it\
Error Threshold - the neural net stops when it goes below this error

## Structure of the Output

Every iteration, if the error decreases the program will print the diagnostic information of the training and the output of the neural network for each training case. The structure of this is as follows:

First, the input values are given in one line, separated by commas. Then, the expected output array is given. Finally, the neural network's output array is given. An example of this is:

    Input:    1,1
    Expected: 1,1,0
    Output:   0.99,0.99,0.005
    
Here, the input is 1, 1. The expected output for this input is 1, 1, 0. What the neural network outputted was 0.99, 0.99, 0.05.

This same structure is repeated for every training case in the training data.