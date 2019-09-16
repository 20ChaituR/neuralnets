To run this neural network, run Main.java. The Neural Network will read the weights from the file "weights.txt". The program will ask for the input values, which are given space-separated in standard input.

**Structure of the Weights File:**

First, the size of the weights matrix between the first two layers is given. The number of rows 
describes the number of activations in the previous layer, and the number of columns describes the 
number of activations in the next layer. After this, each value of the weights matrix is given such 
that each line is a row and the values of each row are space-separated. This same format is repeated 
for each subsequent connectivity layer.

One example of a weights file is:

2 2 \
0.1 0.2 \
0.3 0.4 

2 1 \
0.5 \
0.6 

This weights matrix corresponds to a neural network with 2 input nodes, 2 hidden nodes, and 1 output 
node. There are two connectivity layers which are given by the matrices.