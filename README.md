To run this neural network, run Main.java. The Neural Network will read the weights from the file 
"weights.txt". The program will ask for the input values, which should be space-separated and given in 
standard input.

**Structure of the Weights File:**

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