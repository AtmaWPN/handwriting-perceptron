class Neuron:
    activation = 0.0
    bias = 0.0
    prev_layer = []

    def activate(self):
        self.activation = 0
