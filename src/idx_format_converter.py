import math

import numpy as np


class IDX:
    """
    self.file is the name of the idx file
    self.size is a tuple containing the length of each dimension of the data
    self.data_offset is an int representing the number of bytes before the data
    """
    def __init__(self, filename):
        self.file = filename
        with open(self.file, 'rb') as f:
            f.seek(3)  # first 2 bytes are blank, 3rd represents data type
            dimensions = int.from_bytes(f.read(1), "big")
            size = []
            for i in range(dimensions):
                size.append(int.from_bytes(f.read(4), "big"))
            self.size = tuple(size)
            self.n_area = 1
            for i in self.size:
                self.n_area *= i
            self.data_offset = 4 + 4 * dimensions

    """
    gets a specific byte from the data
    position must be an iterable the same length as self.size containing integers
    returns a byte of data
    """
    def get_val(self, position):
        with open(self.file, 'rb') as f:
            offset = 0
            for i in range(len(self.size)):
                mult = 1
                for j in range(i + 1, len(self.size)):
                    mult *= self.size[j]
                offset += position[i] * mult
            f.seek(self.data_offset + offset)
            return f.read(1)

    """
    gets a specific batch of the data
    batch is the index of the batch of data
    size is the batch size
    returns an ndarray with size as the first dimension, and the remaining n_area as the second
    """
    def get_batch(self, batch, size=200):
        with open(self.file, 'rb') as f:
            count = size * math.floor(self.n_area / self.size[0])
            offset = self.data_offset + batch * size
            return np.frombuffer(f.read(), int, count, offset).reshape((size, math.floor(self.n_area / self.size[0])))
        
    def get_batches(self, size=200):
        batches = list()
        # determine number of batches
        batch_count = math.floor(self.size[0] / size)
        # get batches
        for batch in range(batch_count):
            batches.append(self.get_batch(batch, size=size))
        return batches

    def get_dimensions(self):
        return self.size
