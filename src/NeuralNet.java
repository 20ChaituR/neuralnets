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
 * This class allows one to construct a simply-connected neural network with a variable number of
 * activation layers and number of activations per layer. The connectivity pattern is such that
 * every adjacent layer is fully connected.
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
    * activation layer. Each column is space separated and each row is on the next line. There is
    * a blank line between each layer and this format is repeated for each layer.
    *
    * An example of a weights file would be:
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
   public NeuralNet(String filename) throws IOException {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      String[] splitLine = br.readLine().split(" ");

      numOfLayers = splitLine.length - 1;
      sizeOfLayers = new int[splitLine.length];
      for (int i = 0; i < splitLine.length; i++) {
         sizeOfLayers[i] = Integer.parseInt(splitLine[i]);
      }

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

         for (int i = 0; i < sizeOfLayers[n]; i++) {
            for (int j = 0; j < sizeOfLayers[n + 1]; j++) {
               weights.get(n)[i][j] = new Random().nextGaussian();
            }
         }
      }
   }

   /**
    * Gets the weights for each connectivity layer from the file given by the filename. The format
    * for the weights is as follows: first, the size of each layer is given. Then, for each layer,
    * the matrix for the weights is given. This matrix is such that the number of rows is the number
    * of nodes in the previous activation layer, and the number of columns is the number of nodes in
    * the next activation layer. Each column is space separated and each row is on the next line.
    * There is a blank line between each layer and this format is repeated for each layer.
    *
    * An example of a weights file would be: 2 2 1
    *
    * 0.5 0.5 0.5 0.5
    *
    * 0.3 0.3
    *
    * Here, there are two connectivity layers, shown by the two matrices. The first layer connects
    * two nodes to two nodes, and the second layer connects two nodes to one node.
    *
    * @param filename the name of the file that the weights are stored in
    */
   private void getWeights(String filename) throws IOException {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      String[] splitLine = br.readLine().split(" ");

      numOfLayers = splitLine.length - 1;
      sizeOfLayers = new int[splitLine.length];
      for (int i = 0; i < splitLine.length; i++) {
         sizeOfLayers[i] = Integer.parseInt(splitLine[i]);
      }

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
    * Stores the weights in the file given by the filename. It uses the same format to store the
    * weights as when getting the weights from a file, with each layer separated into blocks of
    * weight matrices.
    *
    * @param filename the name of the file to store the weights in
    */
   public void storeWeights(String filename) throws IOException {

      PrintWriter pw = new PrintWriter(new FileWriter(filename));
      for (int n = 0; n < weights.size(); n++) {
         pw.println(weights.get(n).length + " " + weights.get(n)[0].length);
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
      activations = new ArrayList<>();
      activations.add(input);
      double[] curLayer = input;
      for (int n = 0; n < numOfLayers; n++) {
         double[] nextLayer = new double[sizeOfLayers[n + 1]];
         for (int i = 0; i < sizeOfLayers[n + 1]; i++) {
            for (int j = 0; j < sizeOfLayers[n]; j++) {
               nextLayer[i] += weights.get(n)[j][i] * curLayer[j];
            }
            nextLayer[i] = outputFunction(nextLayer[i]);
         }
         activations.add(nextLayer);
         curLayer = nextLayer;
      }
      return curLayer;
   }

   /**
    * This is the function used to calculate the output of each activation node. Currently, it is
    * just an identity function.
    *
    * @param x the input for the node
    * @return the function applied to the input
    */
   private double outputFunction(double x) {
      return x;
   }

}
