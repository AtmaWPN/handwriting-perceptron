# testing
# idx converter
    # TODO: later
# Perceptron
    # constructor (test size)
    # set_input
    # calculate
    # backprop
    # core dump and load
    # reset

import unittest
from perceptron import Perceptron
import numpy as np

class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.perceptron = Perceptron(IN_dimension=2, OUT_dimension=1, hidden_layer_count=1, hidden_layer_dimension=3)
        self.perceptron.weights = self.perceptron.empty_weights(1)

    def test_constructor(self):
        # layers
        self.assertEqual(3, len(self.perceptron.layers))
        self.assertEqual(2, len(self.perceptron.layers[0]))
        self.assertEqual(3, len(self.perceptron.layers[1]))
        self.assertEqual(1, len(self.perceptron.layers[2]))
        # biases
        self.assertEqual(2, len(self.perceptron.biases))
        self.assertEqual(3, len(self.perceptron.biases[0]))
        self.assertEqual(1, len(self.perceptron.biases[1]))
        # weights
        self.assertEqual(2, len(self.perceptron.weights)) # there should be 2 weight matrices
        self.assertEqual(3, len(self.perceptron.weights[0])) # there should be 3 rows in the first weight matrix
        self.assertEqual(1, len(self.perceptron.weights[1])) # there should be 1 row in the second weight matrix
        self.assertEqual(2, len(self.perceptron.weights[0][0])) # there should be 2 columns in the second weight matrix
        self.assertEqual(3, len(self.perceptron.weights[1][0])) # there should be 3 columns in the second weight matrix

    def test_input(self):
        self.assertTrue(self.perceptron.set_input(np.ones((2, 1))))
        self.assertFalse(self.perceptron.set_input(np.ones((3, 1))))

    def test_calculate(self):
        self.assertEqual(6, self.perceptron.calculate(np.ones((2, 1)))[0][0])

    def test_backprop(self):
        desire = 1 - self.perceptron.calculate(np.ones((2, 1)))

        nabla_bias = self.perceptron.init_biases()

        nabla_weight = self.perceptron.empty_weights(0)
        
        self.perceptron.backpropagate(nabla_bias, nabla_weight, self.perceptron.hidden_layer_count + 1, desire)

        for (bias, bias_change) in zip(self.perceptron.biases, nabla_bias):
            bias += bias_change
        for (weight, weight_change) in zip(self.perceptron.weights, nabla_weight):
            weight += weight_change
        
        new_desire = 1 - self.perceptron.calculate(np.ones((2, 1)))
        self.assert_(abs(int(new_desire)) < abs(int(desire)))

    def test_core_dump(self):
        self.perceptron.calculate(np.ones((2, 1)))
        self.perceptron.core_dump(True)
        self.perceptron.reset()
        self.perceptron.core_load("ai_core_test.npz")
        self.assertEqual(1, self.perceptron.layers[0][0])
        self.assertEqual(1, self.perceptron.layers[0][1])
        self.assertEqual(2, self.perceptron.layers[1][0])
        self.assertEqual(2, self.perceptron.layers[1][1])
        self.assertEqual(2, self.perceptron.layers[1][2])
        self.assertEqual(6, self.perceptron.layers[2][0])

    def test_reset(self):
        self.perceptron.calculate(np.ones(2))
        self.perceptron.reset()
        self.test_constructor()
        self.assertEqual(0, self.perceptron.layers[0][0])
        self.assertEqual(0, self.perceptron.layers[0][1])
        self.assertEqual(0, self.perceptron.layers[1][0])
        self.assertEqual(0, self.perceptron.layers[1][1])
        self.assertEqual(0, self.perceptron.layers[1][2])
        self.assertEqual(0, self.perceptron.layers[2][0])

if __name__ == '__main__':
    unittest.main()

