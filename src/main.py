import sys

from perceptron import Perceptron
import imagePainter
import numpy as np
from idx_format_converter import IDX

"""
deserializes data files into single dimensional lists
"""
def get_image(file, image):
    data = IDX(file)
    data_vector = []
    for row in range(data.get_dimensions[0]):
        for col in range(data.get_dimensions[1]):
            data_vector.append(int.from_bytes(data.get_val((image, row, col)), "big"))
    return data_vector


if __name__ == '__main__':
    # initialize Perceptron
    perceptron = Perceptron(learning_rate=0.000005, IN_dimension=2, OUT_dimension=1, hidden_layer_count=3, hidden_layer_dimension=6)
    # TODO: teach it multiplication
    # set up multiplication dataset
    batch = list()
    labels = list()
    for i in range(1, 13):
        for j in range(1, 13):
            batch.append(np.full((2, 1), np.atleast_2d(np.array([i, j])).T))
            labels.append(np.full((1, 1), i * j))
    for i in range(10):
        print(f"batch {i}")
        perceptron.train(batch, labels)
    for data in batch:
        pass#print(f"Calculated: {self.perceptron.calculate(data)}, Actual: {data[0] * data[1]}")
