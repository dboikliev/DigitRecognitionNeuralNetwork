using DigitRecognition;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace DigitRecognitionNeuralNetwork
{
    class Program
    {
        static void Main()
        {



            var watch = Stopwatch.StartNew();

            Console.WriteLine("Network initialization starts: " + watch.Elapsed);
            var net = new Network(784, 30, 10);
            Console.WriteLine("Network  initialization finished: " + watch.Elapsed);

            Console.WriteLine("Training set starts loading: " + watch.Elapsed);
            using (var reader = new StreamReader(File.Open("./data/train/train.json", FileMode.Open)))
            {
                var json = reader.ReadToEnd();
                var obj = JsonConvert.DeserializeObject<Image[]>(json);
                var data = obj.Select(img =>
                {
                    var imageData = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(img.Pixels.Select(p => p > 0 ? 1D : 0D).ToArray()));
                    double[] labels = new double[10];
                    labels[img.Label] = 1;
                    var labelData = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(labels));

                    return Tuple.Create(imageData, labelData);
                }).OrderBy(_ => Guid.NewGuid());
                var trainingSetInput = data.Take(50000).ToArray();
                var testData = data.Skip(50000).Take(10000).ToArray();

                Console.WriteLine("Training set finished loading: " + watch.Elapsed);
                Control.TryUseNativeMKL();

                Console.WriteLine("Network training starts: " + watch.Elapsed);
                net.Train(trainingSetInput, testData, epochs: 30, miniBatchSize: 10, learningRate: 2);
                Console.WriteLine("Network training finished : " + watch.Elapsed);

            }

            Console.WriteLine("Test set starts loading: " + watch.Elapsed);
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
                Console.WriteLine("Test set finished loading: " + watch.Elapsed);

                Console.WriteLine("Network test starts: " + watch.Elapsed);
                var result = net.Test(input);
                var guesses = result;
                Console.WriteLine("Network test finished: " + watch.Elapsed);
                Console.WriteLine("RESULTS:");

                for (int i = 0; i < output.Length; i++)
                {
                    Console.WriteLine($"Label: { output[i] }, Guessed: { guesses[i] }");
                }
            }
        }

        struct Image
        {
            public int Label { get; set; }
            public double[] Pixels { get; set; }

            public override string ToString()
            {
                return Label.ToString();
            }
        }
    }
}
