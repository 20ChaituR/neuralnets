
/*
 * Created by cravuri on 1/10/20
 */


import sun.tools.jstat.Scale;

import javax.imageio.ImageIO;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;

public class ImageHandler
{

   static int SIZE = 40;

   public static void main(String[] args) throws IOException
   {
      File folder = new File("hands/full-size");
      File[] listOfFiles = folder.listFiles();
      for (File f : listOfFiles)
      {
         BufferedImage image = ImageIO.read(f);

         final int w = image.getWidth();
         final int h = image.getHeight();
         BufferedImage scaledImage = new BufferedImage(SIZE,SIZE, BufferedImage.TYPE_INT_ARGB);
         final AffineTransform at = AffineTransform.getScaleInstance(((double)SIZE) / w, ((double)SIZE) / h);
         final AffineTransformOp ato = new AffineTransformOp(at, AffineTransformOp.TYPE_BICUBIC);
         scaledImage = ato.filter(image, scaledImage);

         ImageIO.write(scaledImage, "JPG", new File("hands/small/" + f.getName()));




//         Image im = ImageIO.read(f);
//         im = im.getScaledInstance(SIZE, SIZE, Image.SCALE_DEFAULT);
//         ImageIO.write((RenderedImage) im, "bmp", new File("hands/small/" + f.getName()));


//         BufferedImage before = ImageIO.read(f);
//         int w = before.getWidth();
//         int h = before.getHeight();
//         System.out.println(f.getName());
//         BufferedImage after = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
//         AffineTransform at = new AffineTransform();
//         at.scale(((double)SIZE) / w, ((double)SIZE) / h);
//         AffineTransformOp scaleOp =
//                 new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
//         after = scaleOp.filter(before, after);
//         System.out.println(after.getWidth());
//         ImageIO.write(after, "bmp", new File("hands/small/" + f.getName()));
      }
   }

}
