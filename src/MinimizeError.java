import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

/**
 * Minimizing
 *
 * @author Chaitanya Ravuri
 * @version September 24, 2019
 */
public class MinimizeError
{
   // the input files are assumed to be in the same directory as this program
   static final String WEIGHTS_FILE = "weights.txt";
   static final String TRAINING_FILE = "trainingData.txt";
   static final String CONFIG_FILE = "config.txt";

   // meta values that configure the training of the neural net
   static int[] layers;
   static double lambdaMult;
   static double learningRate;
   static int epochs;
   static double errorThreshold;

   /**
    * This function reads the training data from a given file, then returns a matrix containing it.
    * This matrix is indexed as trainingData[n][type][i], where n is the test case, type is either 0
    * or 1, 0 for input and 1 for output, and i is the index of the input/output value.
    *
    * The format of the training data is as follows: On the first line, the number of test cases,
    * number of input nodes, number of output nodes are given. Then, on the following lines, for
    * each test case, first the input values are given, space-separated, then the expected output
    * values are given space-separated.
    *
    * An example of a training data file is:
    *
    * 3 2 1
    * 0 0
    * 0
    * 0 1
    * 1
    * 1 0
    * 1
    *
    * In this case, there are 3 test cases, each with 2 input nodes and 1 output node.
    *
    * @param filename the file to read the training data from
    * @return the matrix of training data
    */
   static double[][][] getTrainingData(String filename) throws FileNotFoundException
   {
      Scanner sc = new Scanner(new FileReader(filename));

      int sizeOfData = sc.nextInt();
      int sizeOfInput = sc.nextInt();
      int sizeOfOutput = sc.nextInt();

      double[][][] trainingData = new double[sizeOfData][2][];

      for (int i = 0; i < sizeOfData; i++)
      {
         double[] inputData = new double[sizeOfInput];
         for (int j = 0; j < sizeOfInput; j++)
         {
            inputData[j] = sc.nextDouble();
         }
         trainingData[i][0] = inputData;

         double[] outputData = new double[sizeOfOutput];
         for (int j = 0; j < sizeOfOutput; j++)
         {
            outputData[j] = sc.nextDouble();
         }
         trainingData[i][1] = outputData;
      }

      return trainingData;
   }

   /**
    * This function reads the configuration of the neural net from the config file. The structure
    * of the config file is as follows:
    *
    * First, the size of each layer is given. Each of these sizes are given space-separated.
    *
    * Each of the next lines contain a variable that configures a part of the training:
    *
    * Lambda Multiplier - how much to multiply the learning rate by each epoch
    * Learning Rate - the initial learning rate of the network
    * Epochs - the number of epochs to run
    * Error Threshold - the neural net stops when it goes below this error
    *
    * @param filename the file to read the configuration from
    */
   static void getConfig(String filename) throws FileNotFoundException
   {
      Scanner sc = new Scanner(new FileReader(filename));

      sc.nextLine();
      String[] line = sc.nextLine().split(" ");

      int numLayers = line.length;
      layers = new int[numLayers];
      for (int i = 0; i < numLayers; i++)
      {
         layers[i] = Integer.parseInt(line[i]);
      }

      sc.next();
      lambdaMult = sc.nextDouble();

      sc.next();
      learningRate = sc.nextDouble();

      sc.next();
      epochs = sc.nextInt();

      sc.next();
      errorThreshold = sc.nextDouble();
   }

   /**
    * The main method first loads in the training data and configuration of the neural net. It then repeatedly
    * randomizes the weights and trains them on the training data. It prints the error when it becomes smaller and
    * repeats this loop until the error gets below a threshold value.
    */
   public static void main(String[] args) throws IOException
   {
      // Load the training data from the training file
      double[][][] trainingData = getTrainingData(TRAINING_FILE);

      // Get the configuration of the neural net from the config file
      getConfig(CONFIG_FILE);

      // Create a neural net with the given layer sizes
      NeuralNet nn = new NeuralNet(layers);

      double minError = Double.MAX_VALUE;
      int e = 1;
      while (minError > errorThreshold * errorThreshold)
      {
         // Randomize the weights
         nn.generateWeights();

         // Train with the given configuration
         nn.train(trainingData, learningRate, lambdaMult, epochs, 0);

         // Calculate the error
         double curError = nn.calculateError(trainingData);

         // Print the error and run each of the test cases if error goes down
         if (curError < minError)
         {
            minError = curError;
            nn.storeWeights(WEIGHTS_FILE);
            System.out.println("Iteration " + e + ": Error = " + Math.sqrt(minError));

            System.out.println("In: Out");
            System.out.println("00: " + nn.propagate(new double[]{0, 0})[0]);
            System.out.println("01: " + nn.propagate(new double[]{0, 1})[0]);
            System.out.println("10: " + nn.propagate(new double[]{1, 0})[0]);
            System.out.println("11: " + nn.propagate(new double[]{1, 1})[0]);
            System.out.println();
         }

         e++;
      }
   }

}
