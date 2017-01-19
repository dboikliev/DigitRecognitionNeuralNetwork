using DigitRecognition;
using MathNet.Numerics.LinearAlgebra;
using System;

namespace DigitRecognitionNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var net = new Network(784, 40, 1);
            net.Train(new[] { Tuple.Create(Matrix<double>.Build.Random(784, 1), Matrix<double>.Build.Random(1, 1)) });
        }
    }
}
