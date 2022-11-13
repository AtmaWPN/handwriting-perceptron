import re

import numpy as np
from datetime import datetime

"""
self.layers is a list of activation vectors for each layer, layers[0] is IN, layers[len(layers) - 1] is OUT
self.biases is a list of bias vectors for each layer, biases[0] is first hidden layer, biases[len(biases) - 1] is OUT
self.weights is a list of weight matrices for each layer
"""
class Perceptron:
    layers = list()
    biases = list()
    weights = list()
    generation = 0
    accuracy = 0.0
    training_time = 0

    def __init__(self, IN_dimension=784, OUT_dimension=10, hidden_layer_count=2, hidden_layer_dimension=16):
        # initialize activations
        self.layers.append(np.zeros(IN_dimension))
        for i in range(0, hidden_layer_count):
            self.layers.append(np.zeros(hidden_layer_dimension))
        self.layers.append(np.zeros(OUT_dimension))
        # initialize biases
        for i in range(0, hidden_layer_count):
            self.biases.append(np.zeros(hidden_layer_dimension))
        self.biases.append(np.zeros(OUT_dimension))  # try not using a bias vector for OUT
        # initialize weights
        for i in range(1, len(self.layers)):
            self.weights.append(np.ones((self.layers[i].shape[0], self.layers[i - 1].shape[0])))

    """
    sets the IN layer to a given ndarray
    input_list is an ndarray
    only allows valid lengths of ndarray
    """
    def set_input(self, input_list):
        # TODO: set_input should take an ndarray
        if len(input_list) == len(self.layers[0]):
            self.layers[0] = np.array(input_list)
        else:
            print("invalid input size")

    def calculate(self, input_list):
        self.set_input(input_list)
        for i in range(1, len(self.layers)):
            self.layers[i] = self.weights[i - 1] @ self.layers[i - 1] - self.biases[i - 1]
            np.clip(self.layers[i], 0, None, self.layers[i])
        return self.layers[len(self.layers) - 1]

    """
    performs backprop for every training image in a given batch
    batch is an ndarray
    returns performance and time data
    """
    def train(self, batch):
        # TODO: train()
        # keep track of nabla_influence
        # for each image in batch:
            # calculate
            # get cost (desire)
            # backprop
        # divide nabla_influence by amount of data in batch (average)
        # add nabla_influence to influence
        # record data on performance and time
        pass

    """
    Recursive backpropagation algorithm
    layer is the current layer
    nabla_bias is total_bias_change
    nabla_weight is total_weight_change
    desire is the desire vector for the current layer
    returns the modified influence vector
    """
    def back_propagate(self, nabla_bias, nabla_weight, layer, desire):
        # note: np.sign only works correctly if there are no negative values in the layer vectors
        d_ReLU = np.sign(self.layers[layer])
        nabla_bias[layer - 1] += 2 * d_ReLU * desire
        nabla_weight[layer - 1] += self.layers[layer - 1] @ (2 * d_ReLU * desire)
        if layer == 1:
            return [nabla_bias, nabla_weight]
        else:
            desire = self.weights[layer - 1] @ (2 * d_ReLU * desire)
        return self.back_propagate(nabla_bias, nabla_weight, layer - 1, desire)

    """
    writes a file containing the influence vector
    filename includes the timestamp
    set test to true for a consistent filename
    """
    def core_dump(self, test):
        if test:
            filename = "ai_core_test"
        else:
            # this line turns '2022-11-12 09:47:29.418813' into 'ai_core_2022-11-12_09:47:29.txt'
            filename = f"ai_core_{'_'.join(re.split('[. ]', str(datetime.today()))[:2])}"
        np.savez(filename, layers=self.layers, biases=self.biases, weights=self.weights)

    def core_load(self, filename):
        influence = np.load(filename, allow_pickle=True)
        self.layers = influence['layers']
        self.biases = influence['biases']
        self.weights = influence['weights']

    """
    Use in case of sentience
    """
    def reset(self):
        IN_dimension = np.shape(self.layers[0])[0]
        OUT_dimension = np.shape(self.layers[-1])[0]
        hidden_layer_count = len(self.layers) - 2
        hidden_layer_dimension = np.shape(self.layers[1])[0]

        self.layers = list()
        self.biases = list()
        self.weights = list()
        self.layers.append(np.zeros(IN_dimension))
        for i in range(0, hidden_layer_count):
            self.layers.append(np.zeros(hidden_layer_dimension))
        self.layers.append(np.zeros(OUT_dimension))
        # initialize biases
        for i in range(0, hidden_layer_count):
            self.biases.append(np.zeros(hidden_layer_dimension))
        self.biases.append(np.zeros(OUT_dimension))  # try not using a bias vector for OUT
        # initialize weights
        for i in range(1, len(self.layers)):
            self.weights.append(np.ones((self.layers[i].shape[0], self.layers[i - 1].shape[0])))

    def __str__(self):
        network_str = ""
        for x in range(0, len(self.layers)):
            if x == 0:
                network_str += "IN: "
            elif x == len(self.layers) - 1:
                network_str += "OUT: "
            else:
                network_str += "Hidden Layer " + str(x) + ": "
            for i in range(self.layers[x].shape[0]):
                network_str += str(self.layers[x][i]) + " "
            network_str = network_str.strip() + "\n"
        network_str = network_str.strip()
        return network_str
