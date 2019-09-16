import java.io.IOException;
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

   private static final String WEIGHTS_FILE = "weights.txt";

   public static void main(String[] args) throws IOException {
      Scanner sc = new Scanner(System.in);
      NeuralNet nn = new NeuralNet(WEIGHTS_FILE);

      System.out.println("Is this a Boolean (AND, OR, XOR) Neural Network? (y/n)");

      String ans = sc.next();
      if (ans.charAt(0) == 'y') {
         System.out.println("00: " + nn.propagate(new double[]{0, 0})[0]);
         System.out.println("01: " + nn.propagate(new double[]{0, 1})[0]);
         System.out.println("10: " + nn.propagate(new double[]{1, 0})[0]);
         System.out.println("11: " + nn.propagate(new double[]{1, 1})[0]);
      } else {
         String line = sc.nextLine();
         while (!line.equals("exit")) {
            System.out.println("Give the array for the input values of the Neural Network.\n" +
                    "You should give space-separated doubles for the input array. If you want to exit, print 'exit'.");
            line = sc.nextLine();

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
      }
   }

}
