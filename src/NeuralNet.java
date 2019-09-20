import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
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
public class NeuralNet {

   private final boolean DEBUG = false;

   private int[] sizeOfLayers;                     // number of units in each activation layer
   private int numOfLayers;                        // number of connectivity layers

   private ArrayList<double[][]> weights;          // weights for connections between each layer
   private ArrayList<double[]> activations;        // state of activation for all processing units

   /**
    * Constructor that creates a neural network with the size of each activation layer given. The
    * first layer is the input layer and the last is the output layer. This constructor generates
    * all weights such that the network is fully connected between adjacent layers. All weights are
    * generated randomly using a Gaussian distribution around 0.
    *
    * @param sizeOfLayers the number of processing units in each activation layer
    */
   public NeuralNet(int[] sizeOfLayers) {
      this.sizeOfLayers = sizeOfLayers;
      numOfLayers = sizeOfLayers.length - 1;

      generateWeights();
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
   public NeuralNet(ArrayList<double[][]> weights) {
      this.weights = weights;

      numOfLayers = weights.size();
      sizeOfLayers = new int[weights.size() + 1];
      for (int i = 0; i < numOfLayers; i++) {
         sizeOfLayers[i] = weights.get(i).length;
      }
      sizeOfLayers[numOfLayers] = weights.get(numOfLayers - 1)[0].length;
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
    *    2 2 1
    *
    *    0.5 0.5
    *    0.5 0.5
    *
    *    0.3
    *    0.3
    *
    * Here, there are two connectivity layers, shown by the two matrices. The first layer connects
    * two nodes to two nodes, and the second layer connects two nodes to one node.
    *
    * @param filename the name of the file that the weights are stored in
    */
   public NeuralNet(String filename) throws IOException {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      String[] splitLine = br.readLine().split(" ");

      // Read in size of each layer
      numOfLayers = splitLine.length - 1;
      sizeOfLayers = new int[splitLine.length];
      for (int i = 0; i < splitLine.length; i++) {
         sizeOfLayers[i] = Integer.parseInt(splitLine[i]);
      }

      // Read weights matrix
      weights = new ArrayList<>(numOfLayers);
      for (int n = 0; n < numOfLayers; n++) {
         weights.add(new double[sizeOfLayers[n]][sizeOfLayers[n + 1]]);
         br.readLine();
         for (int i = 0; i < sizeOfLayers[n]; i++) {
            StringTokenizer st = new StringTokenizer(br.readLine());
            for (int j = 0; j < sizeOfLayers[n + 1]; j++) {
               weights.get(n)[i][j] = Double.parseDouble(st.nextToken());
            }
         }
      }
   }

   /**
    * Given the size of each layer and the number of connectivity layers, generates random weights
    * that connect all nodes in each adjacent layer. Weights are randomized using a Gaussian
    * distribution with mean 0 and standard deviation 1.
    */
   private void generateWeights() {
      weights = new ArrayList<>(numOfLayers);
      for (int n = 0; n < numOfLayers; n++) {
         weights.add(new double[sizeOfLayers[n]][sizeOfLayers[n + 1]]);

         // Generates random numbers for each weight
         for (int i = 0; i < sizeOfLayers[n]; i++) {
            for (int j = 0; j < sizeOfLayers[n + 1]; j++) {
               weights.get(n)[i][j] = new Random().nextGaussian();
            }
         }
      }
   }

   /**
    * Stores the weights in the file given by the filename. It uses the same format to store the
    * weights as when getting the weights from a file, with each layer separated into blocks of
    * weight matrices.
    *
    * @param filename the name of the file to store the weights in
    */
   public void storeWeights(String filename) throws IOException {
      PrintWriter pw = new PrintWriter(new FileWriter(filename));

      // Store the size of each layer
      for (int i = 0; i < sizeOfLayers.length; i++) {
         pw.print(sizeOfLayers[i] + " ");
      }
      pw.println("\n");

      // Store weights matrix
      for (int n = 0; n < weights.size(); n++) {
         for (int i = 0; i < weights.get(n).length; i++) {
            for (int j = 0; j < weights.get(n)[0].length; j++) {
               pw.print(weights.get(n)[i][j] + " ");
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
   public double[] propagate(double[] input) {
      activations = new ArrayList<>(numOfLayers + 1);

      activations.add(input);
      double[] curLayer = input;
      for (int n = 0; n < numOfLayers; n++) {
         // calculates the next layer by multiplying the weights by the current layer
         double[] nextLayer = new double[sizeOfLayers[n + 1]];
         for (int i = 0; i < sizeOfLayers[n + 1]; i++) {
            if (DEBUG) System.out.print("DEBUG: a[" + (n+1) + "][" + i + "] = f(");
            for (int j = 0; j < sizeOfLayers[n]; j++) {
               nextLayer[i] += weights.get(n)[j][i] * curLayer[j];
               if (DEBUG) System.out.print("a[" + n + "][" + j + "]*w[" + n + "][" + j + "][" + i + "] + ");
            }
            if (DEBUG) System.out.println(")");
            // applies the output function to the nodes
            nextLayer[i] = outputFunction(nextLayer[i]);
         }
         // adds the next layer to the activation matrix
         activations.add(nextLayer);
         curLayer = nextLayer;
      }

      return curLayer;
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
   public void train(double[][][] trainingData, double[][][] testData, double learningRate, int epochs) {
      double minError = -1;

      for (int e = 1; e <= epochs && learningRate != 0; e++) {
         for (double[][] trainingCase : trainingData) {
            // Get the change in weights for the training case
            ArrayList<double[][]> deltaWeights = getDeltaWeights(trainingCase[0], trainingCase[1]);

            // Add the change in weights times the learning rate to each weight
            for (int n = 0; n < numOfLayers; n++) {
               for (int i = 0; i < sizeOfLayers[n]; i++) {
                  for (int j = 0; j < sizeOfLayers[n + 1]; j++) {
                     weights.get(n)[i][j] += learningRate * deltaWeights.get(n)[i][j];
                  }
               }
            }
         }

         // Calculate the error using the test data
         double curError = 0;
         for (double[][] testCase : testData) {
            double[] output = propagate(testCase[0]);
            double singleError = 0;
            for (int i = 0; i < output.length; i++) {
               singleError += (testCase[1][i] - output[i]) * (testCase[1][i] - output[i]);
            }
            curError += 0.5 * 0.5 * singleError * singleError;
         }

         // Change the learning rate depending on if the error is decreasing or increasing
         if (minError != -1 && curError < minError) {
            learningRate *= 2.0;
         } else if (minError != -1 && curError > minError) {
            learningRate /= 2.0;
         }
         minError = Math.min(minError, curError);

         // Print the current error twenty times
         if (e % (epochs / 20) == 0) {
            System.out.println("Epoch " + e + ": Error = " + Math.sqrt(curError));
         }
      }
   }

   /**
    * Finds the gradient of the error function with respect to each weight, for a given test case.
    * This function assumes the network has two inputs, one hidden layer, and one output.
    *
    * @param input    the input test case to train the network on
    * @param expected the expected output for that test case
    * @return the amount each weight needs to change
    */
   private ArrayList<double[][]> getDeltaWeights(double[] input, double[] expected) {
      double[] output = propagate(input);

      ArrayList<double[][]> deltaWeights = new ArrayList<>();
      for (int n = 0; n < numOfLayers; n++) {
         deltaWeights.add(new double[sizeOfLayers[n]][sizeOfLayers[n + 1]]);
      }

      for (int j = 0; j < sizeOfLayers[1]; j++) {
         deltaWeights.get(1)[j][0] = (expected[0] - output[0]) * outputFunctionPrime(activations.get(2)[0] * (1 - activations.get(2)[0])) *
                 activations.get(1)[j];
      }

      for (int k = 0; k < sizeOfLayers[0]; k++) {
         for (int j = 0; j < sizeOfLayers[1]; j++) {
            deltaWeights.get(0)[k][j] = activations.get(0)[k] * outputFunctionPrime(activations.get(1)[j]) *
                    (expected[0] - output[0]) * outputFunctionPrime(activations.get(2)[0]) * weights.get(1)[j][0];
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
   private double outputFunction(double x) {
      return x;
//      return 1 / (1 + Math.exp(-x));
   }

   /**
    * This returns the derivative of the output function of each activation node evaluated at x.
    *
    * @param x the input for the node
    * @return the derivative of the output function
    */
   private double outputFunctionPrime(double x) {
      return 1.0;
//      return outputFunction(x) * (1 - outputFunction(x));
   }

   // E = 0.5 * (T - F) ^ 2
   private ArrayList<double[][]> backPropagate(double[] input, double[] expected) {
      double[] output = propagate(input);
      double[] delta = new double[output.length];
      for (int i = 0; i < delta.length; i++) {
         delta[i] = expected[i] - output[i];
      }

      ArrayList<double[][]> deltaWeights = new ArrayList<>();
      for (int n = 0; n < numOfLayers; n++) {
         deltaWeights.add(new double[sizeOfLayers[n]][sizeOfLayers[n + 1]]);
      }

      for (int n = numOfLayers - 1; n >= 0; n--) {
         for (int i = 0; i < sizeOfLayers[n]; i++) {
            for (int j = 0; j < sizeOfLayers[n + 1]; j++) {
               deltaWeights.get(n)[i][j] = delta[j] * outputFunctionPrime(activations.get(n + 1)[j]) *
                       activations.get(n)[i];
            }
         }

         double[] newDelta = new double[sizeOfLayers[n]];
         for (int i = 0; i < sizeOfLayers[n]; i++) {
            for (int j = 0; j < sizeOfLayers[n + 1]; j++) {
               newDelta[i] += delta[j] * weights.get(n)[i][j];
            }
         }
         delta = newDelta;
      }

      return deltaWeights;
   }

}
