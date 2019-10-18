To run this neural network, run Main.java. The Neural Network will read the weights from the file 
"weights.txt". The program will ask for the input values, which should be space-separated and given in 
standard input.

You can choose whether or not the network is a Boolean Neural Network. If it is, the network can be 
trained using the data in "trainingData.txt".

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