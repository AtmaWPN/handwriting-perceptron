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

    def get_dimensions(self):
        return self.size
