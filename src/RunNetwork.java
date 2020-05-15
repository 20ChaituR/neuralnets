import java.io.IOException;

public class RunNetwork
{

   public static void main(String[] args) throws IOException
   {
//      for (int i = 1; i <= 5; i++) {
//         ImageWrapper im = new ImageWrapper("hands/small" + i + ".bmp");
//         im.toGrayScale();
//         im.toBMP("hands/gray" + i + ".bmp");
//      }
      // Create network with given weights
      System.out.println("Creating Network...");
      NeuralNet nn = new NeuralNet("weights.txt");

      // Get the five training cases
      System.out.println("Getting Training Data...");
      Main.getConfig(Main.configFile);
      double[][][] trainingData = Main.getTrainingData(Main.trainingFile);

      // For each test case
      for (double[][] testCase : trainingData)
      {
         // Print each input
         StringBuilder printedTestCase = new StringBuilder();
         printedTestCase.append("Input:    ");
         for (int i = 0; i < testCase[0].length; i++)
         {
            printedTestCase.append(testCase[0][i]).append(",");
         }

         // Print the expected output for the test case
         printedTestCase.deleteCharAt(printedTestCase.length() - 1);
         printedTestCase.append("\nExpected: ");
         for (int i = 0; i < testCase[1].length; i++)
         {
            printedTestCase.append(4 * testCase[1][i] + 1).append(",");
         }

         // Print the neural network's output for the test case
         printedTestCase.deleteCharAt(printedTestCase.length() - 1);
         printedTestCase.append("\nOutput:   ");
         double[] output = nn.propagate(testCase[0]);
         for (int i = 0; i < output.length; i++)
         {
            printedTestCase.append(4 * output[i] + 1).append(",");
         }
         printedTestCase.deleteCharAt(printedTestCase.length() - 1);
         System.out.println(printedTestCase + "\n");
      } // for (double[][] testCase : trainingData)
   }

}
