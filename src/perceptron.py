import re

import numpy as np
from datetime import datetime

"""
self.layers is a list of activation vectors for each layer, layers[0] is IN, layers[len(layers) - 1] is OUT
self.biases is a list of bias vectors for each layer, biases[0] is first hidden layer, biases[len(layers) - 1] is OUT
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

    def set_input(self, input_list):
        if len(input_list) == len(self.layers[0]):
            self.layers[0] = np.array(input_list)
        else:
            print("invalid input size")

    def calculate(self):
        for i in range(1, len(self.layers)):
            self.layers[i] = self.weights[i - 1] @ self.layers[i - 1] - self.biases[i - 1]
        return self.layers[len(self.layers) - 1]

    """
    writes a file containing the influence vector
    file has a header containing: training time, generation, and performance
    filename includes the timestamp
    returns the filename
    """
    def core_dump(self):
        # TODO: save influence vector to file
        # this line turns '2022-11-12 09:47:29.418813' into 'ai_core_2022-11-12_09:47:29.txt'
        filename = f"ai_core_{'_'.join(re.split('[. ]', str(datetime.today()))[:2])}.txt"
        f = open(filename, "w")
        f.write(f"Generation: {self.generation}\n")
        f.write(f"Accuracy: {self.accuracy}\n")
        f.write(f"Training Time: {self.training_time}\n")
        f.write(f"{','.join(self.layers)}\n")
        f.close()
        return filename

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
