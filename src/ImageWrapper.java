
/*
 * Created by cravuri on 10/24/19
 */

public class ImageWrapper
{

   public int[][] imageArray;

   double scalingFactor = 1 << 24;

   public ImageWrapper(String fileName)
   {
      imageArray = DibDump.bmpToArray(fileName);
      for (int i = 0; i < imageArray.length; i++) {
         for (int j = 0; j < imageArray[0].length; j++) {
            imageArray[i][j] &= 0x00ffffff;
         }
      }
   }

   public ImageWrapper(double[] image, int height, int width)
   {
      imageArray = new int[height][width];
      for (int i = 0; i < height; i++)
      {
         for (int j = 0; j < width; j++)
         {
            imageArray[i][j] = (int) (image[i * height + j] * scalingFactor);
         }
      }
   }

   public ImageWrapper(int[][] imageArray)
   {
      this.imageArray = imageArray;
   }

   public double[] toDoubleArray()
   {
      double[] imageDoubleArray = new double[imageArray.length * imageArray[0].length];

      for (int r = 0; r < imageArray.length; r++)
      {
         for (int c = 0; c < imageArray[0].length; c++)
         {
            imageDoubleArray[r * imageArray.length + c] = (double) (imageArray[r][c]) / scalingFactor;
         }
      }

      return imageDoubleArray;
   }

   public void toBMP(String fileName)
   {
      DibDump.imageArrayToBMP(imageArray, fileName);
   }

   public int getHeight() {
      return imageArray.length;
   }

   public int getWidth() {
      return imageArray[0].length;
   }

}
