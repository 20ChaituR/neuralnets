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
    * Constructor that creates a neural net with all the weights given. The weights are given in a
    * file, which is then read and put into the weights array. From that, number of layers and size
    * of each layer is determined.
    *
    * @param filename the name of the file that the weights are stored in
    */
   public NeuralNet(String filename) throws IOException {
      getWeights(filename);

      numOfLayers = weights.size();
      sizeOfLayers = new int[weights.size() + 1];
      for (int i = 0; i < numOfLayers; i++) {
         sizeOfLayers[i] = weights.get(i).length;
      }
      sizeOfLayers[numOfLayers] = weights.get(numOfLayers - 1)[0].length;
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
    * for the weights is as follows: first the size of the weights matrix for one layer is given in
    * one line, space-separated. Then each value of the weights matrix is given such that each line
    * is a row of the matrix and each row is space-separated. Then, on the next line, this same
    * structure is repeated for each consecutive layer.
    *
    * An example of a weights file would be:
    * 2 2
    * 0.5 0.5
    * 0.5 0.5
    *
    * 2 1
    * 0.3
    * 0.3
    *
    * Here, there are two connectivity layers, shown by the two matrices. The first layer connects
    * two nodes to two nodes, and the second layer connects two nodes to one node.
    *
    * @param filename the name of the file that the weights are stored in
    */
   private void getWeights(String filename) throws IOException {
      weights = new ArrayList<>();

      BufferedReader br = new BufferedReader(new FileReader(filename));
      String line = br.readLine();
      while (line != null) {
         int numRows = Integer.parseInt(line.split(" ")[0]);
         int numCols = Integer.parseInt(line.split(" ")[1]);

         double[][] layer = new double[numRows][numCols];
         for (int i = 0; i < numRows; i++) {
            line = br.readLine();
            StringTokenizer st = new StringTokenizer(line);
            for (int j = 0; j < numCols; j++) {
               layer[i][j] = Double.parseDouble(st.nextToken());
            }
         }

         weights.add(layer);

         br.readLine();
         line = br.readLine();
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

   private double outputFunction(double x) {
      return x;
   }

}
