import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

/**
 * Error Minimization
 *
 * Repeatedly randomizes the weights and trains the neural network, updating the minimum error each time. The user has
 * the option to override the default configuration, training data, and weights files.
 *
 * In the configuration file, the user can configure the size of each layer of the neural network, the initial learning
 * rate and learning rate multiplier, the number of epochs that will be run, and the minimum error for the network. In
 * the training data file, the user gives the number of training cases, and gives the input and output arrays for each
 * training case. The weights file is where the weights are stored after the neural net completes its training.
 *
 * After the neural network is run, the error and the outputs for each training case is printed.
 *
 * @author Chaitanya Ravuri
 * @version September 24, 2019
 */
public class MinimizeError
{
   // the default input files for the program
   static String weightsFile = "weights.txt";
   static String configFile = "config.txt";
   static String trainingFile = "imageTrainingData.txt";

   // meta values that configure the training of the neural net
   static int sizeOfInput;
   static int sizeOfOutput;
   static int imHeight;
   static int imWidth;
   static int[] layers;
   static double lambdaMult;
   static double learningRate;
   static int epochs;
   static int maxIterations;
   static double errorThreshold;

   static ImageWrapper[][] getTrainingData(String filename) throws IOException
   {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      int numOfTrainingCases = Integer.parseInt(br.readLine());
      ImageWrapper[][] trainingData = new ImageWrapper[numOfTrainingCases][2];
      for (int i = 0; i < numOfTrainingCases; i++)
      {
         String inputFile = br.readLine();
         trainingData[i][0] = new ImageWrapper(inputFile);

         String outputFile = br.readLine();
         trainingData[i][1] = new ImageWrapper(outputFile);
      }

      return trainingData;
   }

   static double scalingFactor = 0x01000000;

   static double[][][] imageTrainingDataToTrainingData(ImageWrapper[][] images) {
      double[][][] trainingData = new double[images.length][images[0].length][];

      for (int i = 0; i < images.length; i++) {
         for (int j = 0; j < images[0].length; j++) {
            trainingData[i][j] = images[i][j].toDoubleArray();
            for (int k = 0; k < trainingData[i][j].length; k++) {
               trainingData[i][j][k] /= scalingFactor;
            }
         }
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
    * Maximum Iterations - the maximum number of times to randomize the weights of the network and retrain it
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
      maxIterations = sc.nextInt();

      sc.next();
      errorThreshold = sc.nextDouble();
   } // static void getConfig(String filename)

   /**
    * The main method first loads in the training data and configuration of the neural net. It then repeatedly
    * randomizes the weights and trains them on the training data. It prints the error when it becomes smaller and
    * repeats this loop until the error gets below a threshold value.
    */
   public static void main(String[] args) throws IOException
   {
      // If the user wants to override the file paths, they can enter in the config, training data, and weights files manually
      Scanner sc = new Scanner(System.in);
      System.out.println("Do you want to override the default file paths? (y/n)");

      String ans = sc.next();
      if (ans.charAt(0) == 'y')
      {
         String defaultResponse = "default";

         System.out.println("What is the file path of the config file? (type " + defaultResponse + " for the default path)");
         ans = sc.next();
         if (!ans.equals(defaultResponse))
         {
            configFile = ans;
         }

         System.out.println("What is the file path of the training data file? (type " + defaultResponse + " for the default path)");
         ans = sc.next();
         if (!ans.equals(defaultResponse))
         {
            weightsFile = ans;
         }

         System.out.println("What is the file path of the weights file? (type " + defaultResponse + " for the default path)");
         ans = sc.next();
         if (!ans.equals(defaultResponse))
         {
            weightsFile = ans;
         }
      }
      else
      {
         System.out.println("\nThe order of outputs for the given input is OR, AND, XOR.\n");
      }

      // Get the configuration of the neural net from the config file
      getConfig(configFile);

      // Load the training data from the training file
      ImageWrapper[][] imageTrainingData = getTrainingData(trainingFile);

      double[][][] trainingData = imageTrainingDataToTrainingData(imageTrainingData);

      // Create a neural net with the given layer sizes
      NeuralNet nn = new NeuralNet(layers);

      double minError = Double.MAX_VALUE;
      int e = 1;
      while (e <= maxIterations && minError > errorThreshold * errorThreshold)
      {
         // Randomize the weights
         nn.generateWeights();

         // Train with the given configuration
         String diagnosticInformation = nn.train(trainingData, learningRate, lambdaMult, epochs);

         // Calculate the error
         double curError = nn.calculateError(trainingData);

         // Print the error and run each of the test cases if error goes down
         if (curError < minError)
         {
            minError = curError;
            nn.storeWeights(weightsFile);
            System.out.println("Iteration " + e);

            System.out.println(diagnosticInformation);

            // For each test case
            for (double[][] testCase : trainingData)
            {
               // Print each input
               StringBuilder printedTestCase = new StringBuilder();
               printedTestCase.append("Input:    ");
               for (int i = 0; i < testCase[0].length; i++)
               {
                  printedTestCase.append((int) testCase[0][i]).append(",");
               }

               // Print the expected output for the test case
               printedTestCase.deleteCharAt(printedTestCase.length() - 1);
               printedTestCase.append("\nExpected: ");
               for (int i = 0; i < testCase[1].length; i++)
               {
                  printedTestCase.append((int) testCase[1][i]).append(",");
               }

               // Print the neural network's output for the test case
               printedTestCase.deleteCharAt(printedTestCase.length() - 1);
               printedTestCase.append("\nOutput:   ");
               double[] output = nn.propagate(testCase[0]);
               for (int i = 0; i < output.length; i++)
               {
                  printedTestCase.append(output[i]).append(",");
               }
               printedTestCase.deleteCharAt(printedTestCase.length() - 1);
               System.out.println(printedTestCase + "\n");
            } // for (double[][] testCase : trainingData)
            System.out.println("\n");
         } // if (curError < minError)

         e++;
      } // while (e <= maxIterations && minError > errorThreshold * errorThreshold)
   } // public static void main(String[] args)

} // public class MinimizeError
