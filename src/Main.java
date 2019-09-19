import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Driver for Neural Network class.
 *
 * Specify the actual neural network in the weights file, "weights.txt". This program can run a
 * neural network with any number of layers and any number of nodes per layer.
 *
 * @author Chaitanya Ravuri
 * @version September 4, 2019
 */
public class Main {

   // "weights.txt" is assumed to be in the same directory as this program
   private static final String WEIGHTS_FILE = "weights.txt";
   private static final String TRAINING_FILE = "trainingData.txt";

   private static double[][][] getTrainingData(String filename) {
      Scanner sc = new Scanner(filename);

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

//      String ans = sc.next();
      String ans = "y";
      if (ans.charAt(0) == 'y') {
         // The program runs the neural net on all of the possible boolean inputs
         ArrayList<double[]> inputs = new ArrayList<>();
         ArrayList<double[]> expected = new ArrayList<>();

         inputs.add(new double[] {0, 0});
         expected.add(new double[] {0});
         inputs.add(new double[] {0, 1});
         expected.add(new double[] {1});
         inputs.add(new double[] {1, 0});
         expected.add(new double[] {1});
         inputs.add(new double[] {1, 1});
         expected.add(new double[] {0});

//         ArrayList<ArrayList<double[]>> trainingData = new ArrayList<>();
//         trainingData.add(inputs);
//         trainingData.add(expected);

         double[][][] trainingData = getTrainingData(TRAINING_FILE);

         int[] layers = new int[] {2, 2, 1};
         NeuralNet nn = new NeuralNet(layers);
         nn.train(trainingData, trainingData,0.01, 100000);

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
