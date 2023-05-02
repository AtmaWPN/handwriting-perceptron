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
    perceptron = Perceptron(learning_rate=0.00000005)
    # set up batches
    #imagePainter.draw_batch("train_set_images", 0)
    train_set = IDX("train_set_images")
    label_data = IDX("train_set_labels")
    batches = train_set.get_batches()
    all_labels = label_data.get_batches()
    print(len(batches))
    print(len(all_labels))
    # train all batches
    i = 0
    csv_file = ""
    for i in range(20):
        for (batch, labels) in zip(batches, all_labels):
            print(f"batch {i}")
            csv_file += str(perceptron.train(batch, labels)) + ","
            i += 1
    perceptron.core_dump(test=True)

    with open("performance_data.csv", "wt") as f:
        f.write(csv_file)

    # test perceptron
    test_set = IDX("test_set_images")
    test_labels = IDX("test_set_labels")
    test_image = test_set.get_batch(0, size=1)[0]
    test_label = test_labels.get_batch(0, size=1)
    print(perceptron.calculate(test_image))
    print(test_label)
    #imagePainter.draw_image("test_set_images", 0)
