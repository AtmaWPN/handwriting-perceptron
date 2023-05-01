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
    # set up batches
    data = IDX("train_set_images")
    label_data = IDX("train_set_labels")
    batches = data.get_batches()
    all_labels = label_data.get_batches()
    print(len(batches))
    print(len(all_labels))
    # train on all batches
    i = 0
    for (batch, labels) in zip(batches, all_labels):
        print(f"batch {i}")
        perceptron.train(batch, labels)
        i += 1