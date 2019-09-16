import java.io.IOException;
import java.util.ArrayList;

public class Main {

   public static final String WEIGHTS_FILE = "weights.txt";

   public static void main(String[] args) throws IOException {
      ArrayList<double[]> inputs = new ArrayList<>();
      ArrayList<double[]> expected = new ArrayList<>();

      inputs.add(new double[]{0, 0});
      expected.add(new double[]{0});
      inputs.add(new double[]{0, 1});
      expected.add(new double[]{1});
      inputs.add(new double[]{1, 0});
      expected.add(new double[]{1});
      inputs.add(new double[]{1, 1});
      expected.add(new double[]{0});

      ArrayList<ArrayList<double[]>> trainingData = new ArrayList<>();
      trainingData.add(inputs);
      trainingData.add(expected);

      int[] layers = new int[]{2, 2, 1};
      NeuralNet nn = new NeuralNet(WEIGHTS_FILE);
//      nn.train(trainingData, trainingData, 0.01, 100000);
      System.out.println("00: " + nn.propagate(new double[]{0, 0})[0]);
      System.out.println("01: " + nn.propagate(new double[]{0, 1})[0]);
      System.out.println("10: " + nn.propagate(new double[]{1, 0})[0]);
      System.out.println("11: " + nn.propagate(new double[]{1, 1})[0]);
      nn.storeWeights(WEIGHTS_FILE);
   }

}
