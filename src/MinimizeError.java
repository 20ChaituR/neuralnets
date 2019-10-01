
/*
 * Created by cravuri on 9/24/19
 */

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

public class MinimizeError
{

   private static double[][][] getTrainingData(String filename) throws FileNotFoundException
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

   private static final double LAMBDA_MULT = 1.001;
   private static final double LEARNING_RATE = 0.001;
   private static final int EPOCHS = 100000;
   private static final double ERROR_THRESHOLD = 0;
   private static final boolean CALCULATE = false;

   public static void main(String[] args) throws IOException
   {
      if (CALCULATE)
      {
         NeuralNet nn = new NeuralNet("weights.txt");

         System.out.println("In: Out");
         System.out.println("00: " + nn.propagate(new double[]{0, 0})[0]);
         System.out.println("01: " + nn.propagate(new double[]{0, 1})[0]);
         System.out.println("10: " + nn.propagate(new double[]{1, 0})[0]);
         System.out.println("11: " + nn.propagate(new double[]{1, 1})[0]);
      }
      else
      {
         double[][][] trainingData = getTrainingData("trainingData.txt");
         int[] layers = new int[]{2, 2, 1};
         NeuralNet nn = new NeuralNet(layers);
         nn.PRINTING_RATE = 0;
         nn.LAMBDA_MULT = LAMBDA_MULT;

         double error = Double.MAX_VALUE;
         int e = 0;
         while (error > ERROR_THRESHOLD * ERROR_THRESHOLD)
         {
            nn.generateWeights();

            nn.train(trainingData, trainingData, LEARNING_RATE, EPOCHS);
            double curError = 0;

            for (double[][] testCase : trainingData)
            {
               double[] output = nn.propagate(testCase[0]);
               double singleError = 0;
               for (int i = 0; i < output.length; i++)
               {
                  singleError += (testCase[1][i] - output[i]) * (testCase[1][i] - output[i]);
               }
               curError += (0.5 * singleError) * (0.5 * singleError);
            }

            if (curError < error)
            {
               error = curError;
               nn.storeWeights("weights.txt");
               System.out.println("Iteration " + e + ": Error = " + Math.sqrt(error));
            }

            e++;
         }
      }
   }

}
