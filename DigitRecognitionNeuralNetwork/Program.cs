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
            var net = new Network(784, 40, 10);
            Console.WriteLine("Network  initialization finished: " + watch.Elapsed);

            Console.WriteLine("Training set starts loading: " + watch.Elapsed);
            using (var reader = new StreamReader(File.Open(@".\data\train\train.json", FileMode.Open)))
            {
                var json = reader.ReadToEnd();
                var obj = JsonConvert.DeserializeObject<Image[]>(json);
                Console.WriteLine(string.Join(", ", obj));

                var trainingSetInput = obj.Select(img =>
                {
                    var imageData = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(img.Pixels));
                    double[] labels = new double[10];
                    labels[img.Label] = 1;
                    var labelData = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(labels));

                    return Tuple.Create(imageData, labelData);
                }).OrderBy(_ => Guid.NewGuid()).Take(10000).ToArray();

                Console.WriteLine("Training set finished loading: " + watch.Elapsed);
                Control.TryUseNativeMKL();
                //var trainingData = new[]
                //{
                //    Vector<double>.Build.DenseOfArray(new [] { 0D }),
                //    Vector<double>.Build.DenseOfArray(new [] { 1D }),
                //};

                //var trainingLabels = new[]
                //{
                //    Vector<double>.Build.DenseOfArray(new[] { 1D }),
                //    Vector<double>.Build.DenseOfArray(new[] { 0D }),
                //};

                //var trainingSet = trainingData.Zip(trainingLabels, (inputs, label) =>
                //    Tuple.Create(Matrix<double>.Build.DenseOfColumnVectors(inputs),
                //                 Matrix<double>.Build.DenseOfColumnVectors(label)))
                //    .ToArray();

                //var t = trainingData.Select(row => Matrix<double>.Build.DenseOfColumnVectors(row));

                //net.Train(new[] { Tuple.Create(Matrix<double>.Build.Random(784, 50000), Matrix<double>.Build.Random(10, 1)) });
                //net.Train(trainingSetInput);
                Console.WriteLine("Network training starts: " + watch.Elapsed);
                net.Train(trainingSetInput, epochs: 30, miniBatchSize: 20, learningRate: 3);
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
                //var test = new[] { Vector<double>.Build.DenseOfArray(new[] { 1D }) };
                //test.Select(t => Matrix<double>.Build.DenseOfColumnVectors(t));
                Console.WriteLine("Network test starts: " + watch.Elapsed);
                var result = net.Test(input);
                var guesses = result;
                Console.WriteLine("Network test finished: " + watch.Elapsed);
                Console.WriteLine("RESULTS:");

                for (int i = 0; i < output.Length; i++)
                {
                    Console.WriteLine($"Label: { output[i] }, Guessed: { guesses[i] }");
                }
                //Console.WriteLine(string.Join<Matrix<double>>(Environment.NewLine, result));
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
