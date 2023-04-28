import re

import numpy as np
from datetime import datetime

"""
self.layers is a list of activation vectors for each layer, layers[0] is IN, layers[len(layers) - 1] is OUT
self.biases is a list of bias vectors for each layer, biases[0] is first hidden layer, biases[len(biases) - 1] is OUT
self.weights is a list of weight matrices for each layer
"""
class Perceptron:
    generation = 0
    accuracy = 0.0
    training_time = 0

    def __init__(self, IN_dimension=784, OUT_dimension=10, hidden_layer_count=2, hidden_layer_dimension=16):
        self.IN_dimension = IN_dimension
        self.OUT_dimension = OUT_dimension
        self.hidden_layer_count = hidden_layer_count
        self.hidden_layer_dimension = hidden_layer_dimension

        # make these initializations into methods
        # initialize activations
        self.layers = list()
        self.layers.append(np.zeros(IN_dimension))
        for i in range(hidden_layer_count):
            self.layers.append(np.zeros(hidden_layer_dimension))
        self.layers.append(np.zeros(OUT_dimension))

        # initialize biases
        self.biases = list()
        for i in range(hidden_layer_count):
            self.biases.append(np.zeros(hidden_layer_dimension))
        self.biases.append(np.zeros(OUT_dimension))  # try not using a bias vector for OUT

        # initialize weights
        self.weights = list()
        for i in range(1, len(self.layers)):
            self.weights.append(np.ones((self.layers[i].shape[0], self.layers[i - 1].shape[0])))

    """
    sets the IN layer to a given ndarray
    input_list is an ndarray
    returns true if input set is valid
    returns false otherwise
    """
    def set_input(self, input_list):
        if len(input_list) == len(self.layers[0]):
            self.layers[0] = np.array(input_list)
            return True
        return False

    """
    calculates the output for a given input
    input_list is an ndarray
    returns the OUT layer
    """
    def calculate(self, input_list):
        self.set_input(input_list) # TODO: either handle errors here instead of in set_input(), or don't nest these methods
        for i in range(1, len(self.layers)):
            self.layers[i] = self.weights[i - 1] @ self.layers[i - 1] - self.biases[i - 1]
            np.clip(self.layers[i], 0, None, self.layers[i]) # ReLU
        return self.layers[len(self.layers) - 1]

    """
    performs backprop for every training image in a given batch
    batch is an ndarray containing the image data of each batch
    labels is an ndarray containing the corresponding labels for the images
    returns performance and time data
    """
    def train(self, batch, labels):
        # TODO: train()
        # keep track of nabla_influence
        nabla_bias = list()
        for i in range(self.hidden_layer_count):
            nabla_bias.append(np.zeros(self.hidden_layer_dimension))
        nabla_bias.append(np.zeros(self.OUT_dimension))

        nabla_weight = list()
        for i in range(1, len(self.layers)):
            nabla_weight.append(np.ones((self.layers[i].shape[0], self.layers[i - 1].shape[0])))
        
        # for each image in batch:
        for (data, label) in zip(batch, labels):
            # calculate
            A_x = self.calculate(data)
            # get cost (desire)
            actual = np.zeros(len(self.layers[len(self.layers) - 1]))
            actual[label - 1] = 1
            
            desire = actual - self.layers[len(self.layers) - 1]
            # backprop
            self.backpropagate(nabla_bias, nabla_weight, self.hidden_layer_count + 1, desire)
            
        # divide nabla_influence by amount of data in batch (average)
        # add nabla_influence to influence
        # record data on performance and time
        pass

    """
    Recursive backpropagation algorithm
    layer is the current layer
    nabla_bias is the total bias change
    nabla_weight is the total weight change
    desire is the desire vector for the current layer
    returns the modified influence vector
    """
    def backpropagate(self, nabla_bias, nabla_weight, layer, desire):
        d_ReLU = np.sign(self.layers[layer])
        # it's nabla_bias[layer - 1] because layer - 1 is the index of the layer in biases (not every layer has biases)
        nabla_bias[layer - 1] += 2 * d_ReLU * desire
        nabla_weight[layer - 1] += self.layers[layer - 1] @ (2 * d_ReLU * desire)
        if layer == 1:
            return [nabla_bias, nabla_weight]
        else:
            desire = self.weights[layer - 1] @ (2 * d_ReLU * desire)
        return self.backpropagate(nabla_bias, nabla_weight, layer - 1, desire)

    """
    writes a file containing the influence vector
    filename includes the timestamp
    set test to true for a consistent filename
    """
    def core_dump(self, test):
        if test:
            filename = "ai_core_test"
        else:
            # this line turns '2022-11-12 09:47:29.418813' into 'ai_core_2022-11-12_09:47:29.npz'
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
        # TODO: maybe store this stuff and make some getters
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
        output = ""
        for layer in range(0, len(self.layers)):
            if layer == 0:
                output += "IN: "
            elif layer == len(self.layers) - 1:
                output += "OUT: "
            else:
                output += "Hidden Layer " + str(layer) + ": "
            for i in range(self.layers[layer].shape[0]):
                output += str(self.layers[layer][i]) + " "
            output = output.strip() + "\n"
        output = output.strip()
        return output
