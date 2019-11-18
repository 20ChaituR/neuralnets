
/*
 * Created by cravuri on 10/24/19
 */

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

public class Main2
{
   // the default input files for the program
   static String weightsFile = "weights.txt";
   static String configFile = "config.txt";
   static String trainingFile = "trainingData.txt";
   static String imageTrainingFile = "imageTrainingData.txt";

   // meta values that configure the training of the neural net
   static int imHeight;
   static int imWidth;
   static int[] layers;
   static double lambdaMult;
   static double learningRate;
   static int epochs;
   static int maxIterations;
   static double errorThreshold;
   static int printingRate;

   static void getConfig(String filename) throws FileNotFoundException
   {
      Scanner sc = new Scanner(new FileReader(filename));

      sc.nextLine();
      String[] line = sc.nextLine().split(" ");

      int numLayers = line.length + 2;
      layers = new int[numLayers];
      for (int i = 1; i < numLayers - 1; i++)
      {
         layers[i] = Integer.parseInt(line[i - 1]);
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

      sc.next();
      printingRate = sc.nextInt();
   } // static void getConfig(String filename)

   static double[][][] getTrainingData(String filename) throws FileNotFoundException
   {
      Scanner sc = new Scanner(new FileReader(filename));

      int sizeOfData = sc.nextInt();
      int sizeOfInput = sc.nextInt();
      int sizeOfOutput = sc.nextInt();

      layers[0] = sizeOfInput;
      layers[layers.length - 1] = sizeOfOutput;

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

   static void loadImages(String inFileName, String outFileName) throws FileNotFoundException
   {
      Scanner sc = new Scanner(new FileReader(inFileName));
      PrintWriter pw = new PrintWriter(outFileName);

      int sizeOfData = sc.nextInt();
      int sizeOfInput = sc.nextInt();
      int sizeOfOutput = sc.nextInt();
      pw.println(sizeOfData + " " + sizeOfInput + " " + sizeOfOutput);

      for (int i = 0; i < sizeOfData; i++)
      {
         ImageWrapper inImage = new ImageWrapper(sc.next());

         imHeight = inImage.getHeight();
         imWidth = inImage.getWidth();

         double[] inArray = inImage.toDoubleArray();
         for (int j = 0; j < inArray.length; j++)
         {
            pw.print(inArray[j] + " ");
         }

         pw.println();

         ImageWrapper outImage = new ImageWrapper(sc.next());
         double[] outArray = outImage.toDoubleArray();
         for (int j = 0; j < outArray.length; j++)
         {
            pw.print(outArray[j] + " ");
         }

         pw.println();
      }

      pw.close();
   }

   public static void main(String[] args) throws IOException
   {
      getConfig(configFile);

//       Image Processing
      loadImages(imageTrainingFile, trainingFile);

      double startTime = System.currentTimeMillis();
      System.out.println("Getting Training Data...");
      double[][][] trainingData = getTrainingData(trainingFile);

      System.out.println((System.currentTimeMillis() - startTime) + ": Creating Network...");
      NeuralNet nn = new NeuralNet(layers);

      // Train with the given configuration
      System.out.println((System.currentTimeMillis() - startTime) + ": Training...");
      String diagnosticInformation = nn.train(trainingData, learningRate, lambdaMult, epochs);

      nn.storeWeights(weightsFile);

      System.out.println(diagnosticInformation);

      // Print out image
      double[] image = nn.propagate(trainingData[0][0]);
      ImageWrapper im = new ImageWrapper(image, imHeight, imWidth);
      im.toBMP("small2.bmp");
   }

}
