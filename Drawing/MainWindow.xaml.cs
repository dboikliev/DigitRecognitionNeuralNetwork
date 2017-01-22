using DigitRecognition;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System;
using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Drawing
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public static object Bitmap { get; private set; }

        public MainWindow()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            int margin = (int)this.DrawingArea.Margin.Left;
            int width = (int)this.DrawingArea.ActualWidth - margin;
            int height = (int)this.DrawingArea.ActualHeight - margin;
            //render ink to bitmap
            RenderTargetBitmap rtb =
            new RenderTargetBitmap(width, height, 96d, 96d, PixelFormats.Default);
            rtb.Render(DrawingArea);
            //save the ink to a memory stream
            BmpBitmapEncoder encoder = new BmpBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(rtb));
            byte[] bitmapBytes;
            using (MemoryStream ms = new MemoryStream())
            {
                encoder.Save(ms);
                //get the bitmap bytes from the memory stream
                ms.Position = 0;
                bitmapBytes = ms.ToArray();
            }


            using (var biasesReader = new StreamReader(File.Open("./trained/biases.txt", FileMode.Open)))
            using (var weightsReader = new StreamReader(File.Open("./trained/weights.txt", FileMode.Open)))
            {
                var biasesJson = biasesReader.ReadToEnd();
                var weightsJson = weightsReader.ReadToEnd();
                if (biasesJson.Length > 0 && weightsJson.Length > 0)
                {
                    var biases = JsonConvert.DeserializeObject<double[][]>(biasesJson);
                    var weights = JsonConvert.DeserializeObject<double[][,]>(weightsJson);
                    var learned = Tuple.Create(biases, weights);

                    var net = new Network(784, 40, 20, 10);

                    net.Load(learned.Item1, learned.Item2);
                    TestImage(net);
                }
            }
        }

        void TestImage(Network net)
        {

            int margin = (int)this.DrawingArea.Margin.Left;
            int width = (int)this.DrawingArea.ActualWidth - margin;
            int height = (int)this.DrawingArea.ActualHeight - margin;
            //render ink to bitmap
            RenderTargetBitmap rtb =
            new RenderTargetBitmap(width, height, 96d, 96d, PixelFormats.Default);
            rtb.Render(DrawingArea);
            //save the ink to a memory stream
            BmpBitmapEncoder encoder = new BmpBitmapEncoder();

            encoder.Frames.Add(BitmapFrame.Create(rtb));

            byte[] bitmapBytes;
            using (MemoryStream ms = new MemoryStream())
            {
                encoder.Save(ms);
                //get the bitmap bytes from the memory stream
                ms.Position = 0;
                bitmapBytes = ms.ToArray();

                var scaleWidth = (int)(DrawingArea.Width * 0.1);
                var scaleHeight = (int)(DrawingArea.Height * 0.1);
                var img = Image.FromStream(ms);
                var bitmap = new Bitmap(28, 28);
                var graph = Graphics.FromImage(bitmap);

                var brush = new SolidBrush(System.Drawing.Color.Black);
                graph.FillRectangle(brush, new RectangleF(0, 0, width, height));
                graph.DrawImage(img, new Rectangle(0, 0, scaleWidth, scaleHeight));


                var imageBytes = new double[784];
                var pixles = bitmap.Size.Height * bitmap.Size.Width;
                for (int i = 0; i < bitmap.Size.Height; i++)
                {
                    for (int j = 0; j < bitmap.Size.Width; j++)
                    {
                        var pixel = ((Bitmap)bitmap).GetPixel(j, i);
                        var grey = 0.29 * pixel.R + 0.59 * pixel.G + 0.12 * pixel.B;
                        imageBytes[i * bitmap.Size.Width + j] = grey / 255D;
                    }
                }

                var input = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(imageBytes));
                var result = net.Test(input);
                MessageBox.Show(result.RowSums().AbsoluteMaximumIndex().ToString(), "Guessed");
            }



            // uncomment for higher quality output
            //graph.InterpolationMode = InterpolationMode.High;
            //graph.CompositingQuality = CompositingQuality.HighQuality;
            //graph.SmoothingMode = SmoothingMode.AntiAlias;



            //for (int i = 0; i < image.Size.Height; i++)
            //{
            //    for (int j = 0; j < image.Size.Width; j++)
            //    {
            //        Console.Write(imageBytes[i * image.Size.Width + j]);
            //    }
            //    Console.WriteLine();
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            DrawingArea.Strokes.Clear();
        }
    }
}