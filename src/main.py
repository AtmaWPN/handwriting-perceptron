import sys

from perceptron import Perceptron
import imagePainter
from idx_format_converter import IDX


def get_image(file, image):
    data = IDX(file)
    data_vector = []
    for row in range(28):
        for col in range(28):
            data_vector.append(int.from_bytes(data.get_val((image, row, col)), "big"))
    return data_vector


if __name__ == '__main__':
    neural_network = Perceptron()
    print(neural_network)
    neural_network.set_input(get_image(sys.argv[1], 0))
    print(neural_network)
    neural_network.calculate(get_image(sys.argv[1], 0))
    print(neural_network)
    neural_network.core_dump(True)
    neural_network.reset()
    print(neural_network)
    neural_network.core_load('ai_core_test.npz')
    print(neural_network)

    # imagePainter.draw_all(sys.argv[1])
