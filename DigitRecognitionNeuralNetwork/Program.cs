using DigitRecognition;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;

namespace DigitRecognitionNeuralNetwork
{
    class Program
    {
        static Stopwatch _watch = Stopwatch.StartNew();

        static void Main()
        {
            Tuple<double[][], double[][,]> learned = null;

            using (var biasesReader = new StreamReader(File.Open("./trained/biases.txt", FileMode.Open)))
            using (var weightsReader = new StreamReader(File.Open("./trained/weights.txt", FileMode.Open)))
            {
                var biasesJson = biasesReader.ReadToEnd();
                var weightsJson = weightsReader.ReadToEnd();
                if (biasesJson.Length > 0 && weightsJson.Length > 0)
                {
                    var biases = JsonConvert.DeserializeObject<double[][]>(biasesJson);
                    var weights = JsonConvert.DeserializeObject<double[][,]>(weightsJson);
                    learned = Tuple.Create(biases, weights);
                }
            }

            Console.WriteLine("Network initialization starts: " + _watch.Elapsed);
            var net = new Network(784, 40, 20, 10);
            Console.WriteLine("Network  initialization finished: " + _watch.Elapsed);

            if (learned == null)
            {
                Console.WriteLine("Training set starts loading: " + _watch.Elapsed);
                learned = Train(net);
                SaveLearnedWeights(learned);
                RunTestSet(net, learned);
            }
            else
            {
                net.Load(learned.Item1, learned.Item2);
                TestImage(net);
            }
        }

        static void SaveLearnedWeights(Tuple<double[][], double[][,]> learned)
        {
            var serializedBiases = JsonConvert.SerializeObject(learned.Item1);

            using (var writer = new StreamWriter(File.Open("./trained/biases.txt", FileMode.OpenOrCreate)))
            {
                writer.Write(serializedBiases);
            }

            var serializedWeights = JsonConvert.SerializeObject(learned.Item2);
            using (var writer = new StreamWriter(File.Open("./trained/weights.txt", FileMode.OpenOrCreate)))
            {
                writer.Write(serializedWeights);
            }

        }

        static void RunTestSet(Network net, Tuple<double[][], double[][,]> learned)
        {
            Console.WriteLine("Test set starts loading: " + _watch.Elapsed);
            using (var reader = new StreamReader(File.Open("./data/train/t10k.json", FileMode.Open)))
            {
                var json = reader.ReadToEnd();
                var images = JsonConvert.DeserializeObject<Image[]>(json);

                var testSet = images.Select(img =>
                {
                    var imageData = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(img.Pixels));
                    double[] labels = new double[10];
                    labels[img.Label] = 1;
                    var labelData = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(labels));
                    return Tuple.Create(imageData, labelData);
                }).ToArray();


                var input = testSet.Select(t => t.Item1).ToArray();
                var output = testSet.Select(t => t.Item2).ToArray();
                Console.WriteLine("Test set finished loading: " + _watch.Elapsed);


                Console.WriteLine("Network test starts: " + _watch.Elapsed);
                var guesses = net.Test(input);
                Console.WriteLine("Network test finished: " + _watch.Elapsed);

                Console.WriteLine("RESULTS:");
                var correct = output.Zip(guesses, (o, g) => o.RowSums().MaximumIndex() - g.RowSums().MaximumIndex())
                    .Where(r => r == 0)
                    .Select(r => 1)
                    .Sum() / (double)output.Length;
                Console.WriteLine(correct * 100);
            }
        }

        static void TestImage(Network net)
        {
            var imageBytes = new double[784];
            var image = Bitmap.FromFile("./seven.png");
            var pixles = image.Size.Height * image.Size.Width;
            for (int i = 0; i < image.Size.Height; i++)
            {
                for (int j = 0; j < image.Size.Width; j++)
                {
                    var pixel = ((Bitmap)image).GetPixel(j, i);
                    var grey = 0.29 * pixel.R + 0.59 * pixel.G + 0.12 * pixel.B;
                    imageBytes[i * image.Size.Width + j] = grey / 255D;
                }
            }

            //for (int i = 0; i < image.Size.Height; i++)
            //{
            //    for (int j = 0; j < image.Size.Width; j++)
            //    {
            //        Console.Write(imageBytes[i * image.Size.Width + j]);
            //    }
            //    Console.WriteLine();
            //}
            var input = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(imageBytes));
            var result = net.Test(input);
            Console.WriteLine(result.RowSums().AbsoluteMaximumIndex());
        }

        static Tuple<double[][], double[][,]> Train(Network net)
        {
            using (var reader = new StreamReader(File.Open("./data/train/train.json", FileMode.Open)))
            {
                var json = reader.ReadToEnd();
                var obj = JsonConvert.DeserializeObject<Image[]>(json);
                var data = obj.Select(img =>
                {
                    var imageData = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(img.Pixels.Select(p => p / 255D).ToArray()));
                    double[] labels = new double[10];
                    labels[img.Label] = 1;
                    var labelData = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(labels));

                    return Tuple.Create(imageData, labelData);
                }).OrderBy(_ => Guid.NewGuid());
                var trainingSetInput = data.Take(50000).ToArray();
                var testData = data.Skip(50000).Take(10000).ToArray();

                Console.WriteLine("Training set finished loading: " + _watch.Elapsed);
                Control.UseNativeMKL();

                Console.WriteLine("Network training starts: " + _watch.Elapsed);
                var learned = net.Train(trainingSetInput, testData, epochs: 30, miniBatchSize: 10, learningRate: 1.9);
                Console.WriteLine("Network training finished : " + _watch.Elapsed);
                return learned;
            }
        }

        struct Image
        {
            public int Label { get; set; }
            public double[] Pixels { get; set; }

            internal static void FromFile(string v)
            {
                throw new NotImplementedException();
            }

            public override string ToString()
            {
                return Label.ToString();
            }
        }
    }
}
