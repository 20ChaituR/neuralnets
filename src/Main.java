import java.io.IOException;
import java.util.ArrayList;

public class Main {

   private static final String WEIGHTS_FILE = "weights.txt";

   public static void main(String[] args) throws IOException {
      NeuralNet nn = new NeuralNet(WEIGHTS_FILE);
      System.out.println("00: " + nn.propagate(new double[]{0, 0})[0]);
      System.out.println("01: " + nn.propagate(new double[]{0, 1})[0]);
      System.out.println("10: " + nn.propagate(new double[]{1, 0})[0]);
      System.out.println("11: " + nn.propagate(new double[]{1, 1})[0]);
   }

}
