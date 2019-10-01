import java.io.*;
import java.util.Random;
import java.util.StringTokenizer;

/**
 * Neural Network
 *
 * This class allows one to construct a neural network with a variable number of activation layers
 * and number of activations per layer. The connectivity pattern is such that every adjacent layer
 * is fully connected.
 *
 * This neural network can be constructed by either passing a file which contains the full weights
 * matrix, or by passing the size of each layer, which will then construct the weights matrix with
 * random values. In addition, these weights can be stored into a file. The network is run by
 * calling the propagate function, which calculates the output of the network from the given input.
 *
 * @author Chaitanya Ravuri
 * @version September 4, 2019
 */
public class NeuralNet
{

   private final boolean DEBUG = false;

   public double LAMBDA_MULT = 1.01;
   public double PRINTING_RATE = 20;

   private int[] sizeOfLayers;            // number of units in each activation layer
   private int numOfLayers;               // number of connectivity layers

   private double[][][] weights;          // weights for connections between each layer
   private double[][] activations;        // state of activation for all processing units

   /**
    * Constructor that creates a neural network with the size of each activation layer given. The
    * first layer is the input layer and the last is the output layer. This constructor generates
    * all weights such that the network is fully connected between adjacent layers. Weights are
    * randomized using a Gaussian distribution with mean 0 and standard deviation 1.
    *
    * @param sizeOfLayers the number of processing units in each activation layer
    */
   public NeuralNet(int[] sizeOfLayers)
   {
      this.sizeOfLayers = sizeOfLayers;
      numOfLayers = sizeOfLayers.length - 1;

      generateWeights();
      createActivations();
   }

   /**
    * Given the size of each layer and the number of connectivity layers, generates random weights
    * that connect all nodes in each adjacent layer. Weights are randomized using a Gaussian
    * distribution with mean 0 and standard deviation 1.
    */
   public void generateWeights()
   {
      weights = new double[numOfLayers][][];
      for (int n = 0; n < numOfLayers; n++)
      {
         weights[n] = new double[sizeOfLayers[n]][sizeOfLayers[n + 1]];

         // Generates random numbers for each weight
         for (int i = 0; i < sizeOfLayers[n]; i++)
         {
            for (int j = 0; j < sizeOfLayers[n + 1]; j++)
            {
               weights[n][i][j] = new Random().nextGaussian();
            }
         }
      }
   }

   /**
    * Constructor that creates a neural network with all the weights given. The weights are assumed
    * to be for connections between each adjacent layer of the network. The weights array has three
    * indices: weights[n][i][j]. n describes what connectivity layer the weight is in, i describes
    * what node in the previous layer the weight is connected to and j describes what node in the
    * next layer the weight is connected to.
    *
    * @param weights the weights of each connection in the network
    */
   public NeuralNet(double[][][] weights)
   {
      this.weights = weights;

      numOfLayers = weights.length;
      sizeOfLayers = new int[weights.length + 1];
      for (int i = 0; i < numOfLayers; i++)
      {
         sizeOfLayers[i] = weights[i].length;
      }
      sizeOfLayers[numOfLayers] = weights[numOfLayers - 1][0].length;

      createActivations();
   }

   /**
    * Constructor that creates a neural net with all the weights given. The format for the weights
    * is as follows: first, the size of each layer is given. Then, for each layer, the matrix for
    * the weights is given. This matrix is such that the number of rows is the number of nodes in
    * the previous activation layer, and the number of columns is the number of nodes in the next
    * activation layer. Each column is space separated and each row is on the next line. There is a
    * blank line between each layer and this format is repeated for each layer.
    *
    * An example of a weights file would be:
    *
    * 2 2 1
    *
    * 0.5 0.5
    * 0.5 0.5
    *
    * 0.3
    * 0.3
    *
    * Here, there are two connectivity layers, shown by the two matrices. The first layer connects
    * two nodes to two nodes, and the second layer connects two nodes to one node.
    *
    * @param filename the name of the file that the weights are stored in
    */
   public NeuralNet(String filename) throws IOException
   {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      String[] splitLine = br.readLine().split(" ");

      // Read in size of each layer
      numOfLayers = splitLine.length - 1;
      sizeOfLayers = new int[splitLine.length];
      for (int i = 0; i < splitLine.length; i++)
      {
         sizeOfLayers[i] = Integer.parseInt(splitLine[i]);
      }

      // Read weights matrix
      weights = new double[numOfLayers][][];
      for (int n = 0; n < numOfLayers; n++)
      {
         weights[n] = new double[sizeOfLayers[n]][sizeOfLayers[n + 1]];
         br.readLine();

         for (int i = 0; i < sizeOfLayers[n]; i++)
         {
            StringTokenizer st = new StringTokenizer(br.readLine());  // Each line is a different row of weights
            for (int j = 0; j < sizeOfLayers[n + 1]; j++)
            {
               weights[n][i][j] = Double.parseDouble(st.nextToken()); // Weights are space-separated on each line
            }
         }
      }

      createActivations();
   }

   /**
    * Creates an empty activations matrix given that the array sizeOfLayers is already created. If
    * so, the activations matrix will be a jagged matrix with each row having a number of columns
    * equal to the size of that layer.
    */
   private void createActivations()
   {
      activations = new double[numOfLayers + 1][];     // There is one more activation layer than connectivity layers
      for (int n = 0; n < sizeOfLayers.length; n++)
      {
         activations[n] = new double[sizeOfLayers[n]]; // Each activation layer's size is given in sizeOfLayers
      }
   }

   /**
    * Stores the weights in the file given by the filename. It uses the same format to store the
    * weights as when getting the weights from a file, with each layer separated into blocks of
    * weight matrices.
    *
    * @param filename the name of the file to store the weights in
    */
   public void storeWeights(String filename) throws IOException
   {
      PrintWriter pw = new PrintWriter(new FileWriter(filename));

      // Store the size of each layer
      for (int i = 0; i < sizeOfLayers.length; i++)
      {
         pw.print(sizeOfLayers[i] + " ");
      }
      pw.println("\n");

      // Store weights matrix
      for (int n = 0; n < weights.length; n++)
      {
         for (int i = 0; i < weights[n].length; i++)
         {
            for (int j = 0; j < weights[n][0].length; j++)
            {
               pw.print(weights[n][i][j] + " ");
            }
            pw.println();
         }
         pw.println();
      }
      pw.close();
   }

   /**
    * Given the activations for all input nodes, this function propagates those inputs through the
    * neural net by multiplying each activation layer by the corresponding weights matrix and
    * applying the output function. This is repeated until the values for the output layer are
    * found. This output array is returned.
    *
    * @param input the values for the activation of all input units
    * @return the array of activations for the output units
    */
   public double[] propagate(double[] input)
   {
      activations[0] = input;
      for (int n = 0; n < numOfLayers; n++)
      {
         // calculates the next layer by multiplying the weights by the current layer
         for (int i = 0; i < sizeOfLayers[n + 1]; i++)
         {
            activations[n + 1][i] = 0;
            if (DEBUG) System.out.print("DEBUG: a[" + (n + 1) + "][" + i + "] = f(");
            for (int j = 0; j < sizeOfLayers[n]; j++)
            {
               activations[n + 1][i] += weights[n][j][i] * activations[n][j];
               if (DEBUG)
                  System.out.print("a[" + n + "][" + j + "]*w[" + n + "][" + j + "][" + i + "] + ");
            }
            if (DEBUG) System.out.println(")");

            // applies the output function to the nodes
            activations[n + 1][i] = outputFunction(activations[n + 1][i]);
         }
      }

      return activations[numOfLayers];
   }

   /**
    * Trains the neural network with the given training data and calculates the error with the test
    * data. The learning rate of the network starts at the given learning rate, and increases or
    * decreases depending on the error. Training runs for a given number of epochs or until the
    * learning rate goes to 0.
    *
    * @param trainingData the data that is used to train the network
    * @param testData     the data that is used to calculate the error for the network
    * @param learningRate the initial learning rate of the network
    * @param epochs       the number of epochs that training will run for
    */
   public void train(double[][][] trainingData, double[][][] testData, double learningRate, int epochs)
   {
      double minError = -1;

      for (int e = 1; e <= epochs && learningRate != 0; e++)
      {
         double[][][] deltaWeights = new double[numOfLayers][][];
         for (int n = 0; n < numOfLayers; n++)
         {
            deltaWeights[n] = new double[sizeOfLayers[n]][sizeOfLayers[n + 1]];
         }

         for (double[][] trainingCase : trainingData)
         {
            // find how much the weights need to change for each training case
            double[][][] caseDeltaWeights = getDeltaWeights(trainingCase[0], trainingCase[1]);
            for (int n = 0; n < numOfLayers; n++)
            {
               for (int i = 0; i < sizeOfLayers[n]; i++)
               {
                  for (int j = 0; j < sizeOfLayers[n + 1]; j++)
                  {
                     deltaWeights[n][i][j] += caseDeltaWeights[n][i][j];
                  }
               }
            }
         }

         for (int n = 0; n < numOfLayers; n++)
         {
            for (int i = 0; i < sizeOfLayers[n]; i++)
            {
               for (int j = 0; j < sizeOfLayers[n + 1]; j++)
               {
                  weights[n][i][j] += learningRate * deltaWeights[n][i][j];
               }
            }
         }

         // Calculate the error using the test data
         double curError = 0;
         for (double[][] testCase : testData)
         {
            double[] output = propagate(testCase[0]);
            double singleError = 0;
            for (int i = 0; i < output.length; i++)
            {
               singleError += (testCase[1][i] - output[i]) * (testCase[1][i] - output[i]);
            }
            curError += (0.5 * singleError) * (0.5 * singleError);
         }

         // Change the learning rate depending on if the error is decreasing or increasing
         if (minError != -1 && curError < minError)
         {
            learningRate *= LAMBDA_MULT;
            minError = curError;
         }
         else if (minError != -1 && curError >= minError)
         {
            for (int n = 0; n < numOfLayers; n++)
            {
               for (int i = 0; i < sizeOfLayers[n]; i++)
               {
                  for (int j = 0; j < sizeOfLayers[n + 1]; j++)
                  {
                     weights[n][i][j] -= learningRate * deltaWeights[n][i][j];
                  }
               }
            }
            learningRate /= LAMBDA_MULT;
         }
         else
         {
            minError = curError;
         }

         // Print the current error
         if (PRINTING_RATE > 0 && e % (epochs / PRINTING_RATE) == 0)
         {
            System.out.println("Epoch " + e + ": Error = " + Math.sqrt(minError));
         }
      }
   }

   /**
    * Finds the gradient of the error function with respect to each weight, for a given test case.
    * This function assumes the network has two inputs, one hidden layer, and one output.
    *
    * @param input    the input test case to train the network on
    * @param expected the expected output for that test case
    */
   private double[][][] getDeltaWeights(double[] input, double[] expected)
   {
      double[] output = propagate(input);

      double[][][] deltaWeights = new double[numOfLayers][][];
      for (int n = 0; n < numOfLayers; n++)
      {
         deltaWeights[n] = new double[sizeOfLayers[n]][sizeOfLayers[n + 1]];
      }

      for (int j = 0; j < sizeOfLayers[1]; j++)
      {
         deltaWeights[1][j][0] = (expected[0] - output[0]) * outputFunctionPrime(activations[2][0]) *
                 activations[1][j];
      }

      for (int k = 0; k < sizeOfLayers[0]; k++)
      {
         for (int j = 0; j < sizeOfLayers[1]; j++)
         {
            deltaWeights[0][k][j] = activations[0][k] * outputFunctionPrime(activations[1][j]) *
                    (expected[0] - output[0]) * outputFunctionPrime(activations[2][0]) * weights[1][j][0];
         }
      }

      return deltaWeights;
   }

   /**
    * This is the function used to calculate the output of each activation node.
    *
    * @param x the input for the node
    * @return the function applied to the input
    */
   private double outputFunction(double x)
   {
//      return x;
      return 1 / (1 + Math.exp(-x));
   }

   /**
    * This returns the derivative of the output function of each activation node evaluated at x.
    *
    * @param x the input for the node
    * @return the derivative of the output function
    */
   private double outputFunctionPrime(double x)
   {
//      return 1.0;
      return x * (1 - x);
//      return outputFunction(x) * (1 - outputFunction(x));
   }

   // E = 0.5 * (T - F) ^ 2
   private double[][][] backPropagate(double[] input, double[] expected)
   {
      double[] output = propagate(input);
      double[] delta = new double[output.length];
      for (int i = 0; i < delta.length; i++)
      {
         delta[i] = expected[i] - output[i];
      }

      double[][][] deltaWeights = new double[numOfLayers][][];
      for (int n = 0; n < numOfLayers; n++)
      {
         deltaWeights[n] = new double[sizeOfLayers[n]][sizeOfLayers[n + 1]];
      }

      for (int n = numOfLayers - 1; n >= 0; n--)
      {
         for (int i = 0; i < sizeOfLayers[n]; i++)
         {
            for (int j = 0; j < sizeOfLayers[n + 1]; j++)
            {
               deltaWeights[n][i][j] = delta[j] * outputFunctionPrime(activations[n + 1][j]) *
                       activations[n][i];
            }
         }

         double[] newDelta = new double[sizeOfLayers[n]];
         for (int i = 0; i < sizeOfLayers[n]; i++)
         {
            for (int j = 0; j < sizeOfLayers[n + 1]; j++)
            {
               newDelta[i] += delta[j] * weights[n][i][j];
            }
         }
         delta = newDelta;
      }

      return deltaWeights;
   }

}
