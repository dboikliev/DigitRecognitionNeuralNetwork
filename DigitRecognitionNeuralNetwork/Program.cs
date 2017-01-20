using DigitRecognition;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Diagnostics;
using System.Linq;

namespace DigitRecognitionNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            //using (var reader = new StreamReader(File.Open(@".\data\train\train.json", FileMode.Open)))
            //{
            //var json = reader.ReadToEnd();
            //var obj = JsonConvert.DeserializeObject<Image[]>(json);
            //Console.WriteLine(string.Join(", ", obj));

            //var trainingSetInput = obj.Select(img =>
            //{
            //    var imageData = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(img.Pixels));
            //    double[] labels = new double[10];
            //    labels[img.Label] = 1;
            //    var labelData = Matrix<double>.Build.DenseOfColumnVectors(Vector<double>.Build.DenseOfArray(labels));

            //    return Tuple.Create(imageData, labelData);
            //}).ToArray();
            Console.WriteLine();
            var trainingData = new Vector<double>[]
            {
                    Vector<double>.Build.DenseOfArray(new [] { 0D, 0D }),
                    Vector<double>.Build.DenseOfArray(new [] { 0D, 1D }),
                    Vector<double>.Build.DenseOfArray(new [] { 1D, 0D }),
                    Vector<double>.Build.DenseOfArray(new [] { 1D, 1D }),
            };

            var trainingLabels = new[]
            {
                    Vector<double>.Build.DenseOfArray(new[] { 0D }),
                    Vector<double>.Build.DenseOfArray(new[] { 1D }),
                    Vector<double>.Build.DenseOfArray(new[] { 1D }),
                    Vector<double>.Build.DenseOfArray(new[] { 0D }),
                };

            var trainingSet = trainingData.Zip(trainingLabels, (inputs, label) =>
            {
                return Tuple.Create(Matrix<double>.Build.DenseOfColumnVectors(inputs), Matrix<double>.Build.DenseOfColumnVectors(label));
            }).ToArray();

            var watch = new Stopwatch();
            watch.Start();

            var net = new Network(2, 40, 1);
            Console.WriteLine("Network created: " + watch.Elapsed);
            //var t = trainingData.Select(row => Matrix<double>.Build.DenseOfColumnVectors(row));

            //net.Train(new[] { Tuple.Create(Matrix<double>.Build.Random(784, 50000), Matrix<double>.Build.Random(10, 1)) });
            //net.Train(trainingSetInput);
            net.Train(trainingSet, epochs: 50000, miniBatchSize: 4, learningRate: 0.005);
            Console.WriteLine("Network trained: " + watch.Elapsed);

            var test = new[]
            {
                Vector<double>.Build.DenseOfArray(new[] { 0D, 0D }),
                Vector<double>.Build.DenseOfArray(new[] { 1D, 0D }),
                Vector<double>.Build.DenseOfArray(new[] { 0D, 1D }),
                Vector<double>.Build.DenseOfArray(new[] { 1D, 1D }),
            };
            //test.Select(t => Matrix<double>.Build.DenseOfColumnVectors(t));
            Console.WriteLine("Network test begins: " + watch.Elapsed);
            var result = net.Test(test.Select(t => Matrix<double>.Build.DenseOfColumnVectors(t)).ToArray());
            Console.WriteLine(string.Join<Matrix<double>>(Environment.NewLine, result));
            Console.WriteLine("Network test finisehd: " + watch.Elapsed);
            //}
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
