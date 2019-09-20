import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

/**
 * Driver for Neural Network class.
 *
 * Specify the actual neural network in the weights file, "weights.txt", and specify the training
 * data in "trainingData.txt". This program can run a neural network with any number of layers and
 * any number of nodes per layer.
 *
 * @author Chaitanya Ravuri
 * @version September 4, 2019
 */
public class Main {

   // the input files are assumed to be in the same directory as this program
   private static final String WEIGHTS_FILE = "weights.txt";
   private static final String TRAINING_FILE = "trainingData.txt";

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
    *    3 2 1
    *    0 0
    *    0
    *    0 1
    *    1
    *    1 0
    *    1
    *
    * In this case, there are 3 test cases, each with 2 input nodes and 1 output node.
    *
    * @param filename the file to read the training data from
    * @return the matrix of training data
    */
   private static double[][][] getTrainingData(String filename) throws FileNotFoundException {
      Scanner sc = new Scanner(new FileReader(filename));

      int sizeOfData = sc.nextInt();
      int sizeOfInput = sc.nextInt();
      int sizeOfOutput = sc.nextInt();

      double[][][] trainingData = new double[sizeOfData][2][];

      for (int i = 0; i < sizeOfData; i++) {
         double[] inputData = new double[sizeOfInput];
         for (int j = 0; j < sizeOfInput; j++) {
            inputData[j] = sc.nextDouble();
         }
         trainingData[i][0] = inputData;

         double[] outputData = new double[sizeOfOutput];
         for (int j = 0; j < sizeOfOutput; j++) {
            outputData[j] = sc.nextDouble();
         }
         trainingData[i][1] = outputData;
      }

      return trainingData;
   }

   public static void main(String[] args) throws IOException {
      Scanner sc = new Scanner(System.in);

      System.out.println("Is this a Boolean (2 Inputs, 1 Output) Neural Network? (y/n)");

      String ans = sc.next();
//      String ans = "y";
      if (ans.charAt(0) == 'y') {
         // The program runs the neural net on all of the possible boolean inputs
         double[][][] trainingData = getTrainingData(TRAINING_FILE);

         int[] layers = new int[]{2, 2, 1};
         NeuralNet nn = new NeuralNet(layers);
         nn.train(trainingData, trainingData, 0.01, 100000);

         System.out.println("In: Out");
         System.out.println("00: " + nn.propagate(new double[]{0, 0})[0]);
         System.out.println("01: " + nn.propagate(new double[]{0, 1})[0]);
         System.out.println("10: " + nn.propagate(new double[]{1, 0})[0]);
         System.out.println("11: " + nn.propagate(new double[]{1, 1})[0]);

         nn.storeWeights("weights.txt");
      } else {
         NeuralNet nn = new NeuralNet(WEIGHTS_FILE);

         // The program repeatedly asks for an input array, and propagates it through the network
         String line = sc.nextLine();
         while (!line.equals("exit")) {
            System.out.println("Give the array for the input values of the Neural Network.\n" +
                    "You should give space-separated doubles for the input array. For example: '0.6 1.0'\n" +
                    "If you want to exit, print 'exit'.");
            line = sc.nextLine();

            if (!line.equals("exit")) {
               String[] splitLine = line.split(" ");
               double[] inputs = new double[splitLine.length];
               for (int i = 0; i < inputs.length; i++) {
                  inputs[i] = Double.parseDouble(splitLine[i]);
               }

               double[] outputs = nn.propagate(inputs);
               System.out.println("Outputs:");
               for (int i = 0; i < outputs.length; i++) {
                  System.out.print(outputs[i] + " ");
               }
               System.out.println("\n");
            }
         } // while (!line.equals("exit"))
      }
   }

}
