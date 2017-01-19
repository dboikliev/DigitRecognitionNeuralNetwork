﻿using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using static System.Linq.Enumerable;

namespace DigitRecognition
{
    class Network
    {
        private readonly int _layersCount;
        private readonly int[] _sizes;
        private readonly Matrix<double>[] _biases;
        private readonly Matrix<double>[] _weights;

        public Network(params int[] sizes)
        {
            _layersCount = sizes.Length;
            _sizes = sizes;
            _biases = sizes.Skip(1).Select(size => Matrix<double>.Build.Random(size, 1)).ToArray();
            _weights = sizes.Take(sizes.Length - 1).Zip(sizes.Skip(1), (second, first) =>
            {
                Console.WriteLine($"{ first} {second}");
                return Matrix<double>.Build.Random(first, second);
            }).ToArray();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double Sigmoid(double z)
        {
            return SpecialFunctions.Logistic(z);
        }

        private Matrix<double> FeedForward(Matrix<double> a)
        {
            a = a.Transpose();
            var biasWeightPairs = _biases.Zip(_weights, (bias, weight) => new { bias, weight });
            foreach (var tuple in biasWeightPairs)
            {
                a *= tuple.weight;
                a += tuple.bias.Transpose();
                a = Sigmoid(a);
            }
            return a;
        }

        private Matrix<double> Sigmoid(Matrix<double> matrix)
        {
            return matrix.Map(Sigmoid);
        }

        public void Train(Tuple<Matrix<double>, Matrix<double>>[] trainingSet, int epochs = 30, int miniBatchSize = 20, double learningRate = 0.9)
        {
            for (int i = 0; i < trainingSet.Length; i += miniBatchSize)
            {
                var miniBatch = trainingSet.Skip(i).Take(miniBatchSize).ToArray();
                UpdateMiniBatch(miniBatch, learningRate);
            }
            //UpdateMiniBatch(null, 1);
        }

        private void UpdateMiniBatch(Tuple<Matrix<double>, Matrix<double>>[] miniBatch, double learningRate)
        {
            var nablaB = _biases
                .Select(b => Matrix<double>.Build.Dense(b.RowCount, b.ColumnCount, 0))
                .ToArray();

            var nablaW = _weights
                .Select(w => Matrix<double>.Build.Dense(w.RowCount, w.ColumnCount, 0))
                .ToArray();
            Console.WriteLine(string.Join<Matrix<double>>(Environment.NewLine, nablaB));
            Console.WriteLine(string.Join<Matrix<double>>(Environment.NewLine, nablaW));

            for (int i = 0; i < miniBatch.Length; i++)
            {
                var result = Backprop(miniBatch[i].Item1, miniBatch[i].Item2);
                nablaB = nablaB
                    .Zip(result.Item1, (nb, dnb) => new { nb, dnb })
                    .Select(pair => pair.nb + pair.dnb).ToArray();

                nablaW = nablaW
                    .Zip(result.Item2, (nw, dnw) => new { nw, dnw })
                    .Select(pair => pair.nw + pair.dnw).ToArray();
            }

            _weights.Zip(nablaW, (w, nw) => new { w, nw })
                .Select(pair => pair.w - (learningRate / miniBatch.Length) * pair.nw);

            _biases.Zip(nablaW, (b, nb) => new { b, nb })
                .Select(pair => pair.b - (learningRate / miniBatch.Length) * pair.nb);
        }

        private Tuple<Matrix<double>[], Matrix<double>[]> Backprop(Matrix<double> input, Matrix<double> output)
        {
            var nablaB = _biases
               .Select(b => Matrix<double>.Build.Dense(b.RowCount, b.ColumnCount, 0))
               .ToArray();

            var nablaW = _weights
                .Select(w => Matrix<double>.Build.Dense(w.RowCount, w.ColumnCount, 0))
                .ToArray();

            var activation = input;

            var activations = new List<Matrix<double>> { input };

            var zs = new List<Matrix<double>>();

            var biasWeightPairs = _biases.Zip(_weights, (bias, weight) => new { bias, weight });
            //activation = activation.Transpose();
            foreach (var pair in biasWeightPairs)
            {
                var z = pair.weight.Multiply(activation) + pair.bias;
                zs.Add(z);
                activation = Sigmoid(z);
                activations.Add(activation);
            }

            var delta = CostDerivative(activations[activations.Count - 1], output) * Sigmoid(zs[zs.Count - 1]);
            Console.WriteLine(delta);
            nablaB[nablaB.Length - 1] = delta;
            nablaW[nablaW.Length - 1] = (delta * activations[activations.Count - 2].Transpose());
            Console.WriteLine(string.Join<Matrix<double>>(Environment.NewLine, nablaW));

            var range = Enumerable.Range(2, _layersCount - 2);
            foreach (var i in range)
            {
                var z = zs[zs.Count - i];
                var sp = SigmoidPrime(z);
                delta = (_weights[_weights.Length - i + 1].Transpose() * delta).PointwiseMultiply(sp);
                nablaB[nablaB.Length - i] = delta;
                nablaW[nablaW.Length - i] = (delta * activations[activations.Count - i - 1].Transpose());
            }

            return Tuple.Create(nablaB, nablaW);
        }

        private Matrix<double> SigmoidPrime(Matrix<double> z)
        {
            return Sigmoid(z).PointwiseMultiply(1 - Sigmoid(z));
        }

        private Matrix<double> CostDerivative(Matrix<double> outputActivations, Matrix<double> actual)
        {
            return outputActivations - actual;
        }
    }
}