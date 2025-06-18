import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs + 1) * 0.01
        self.output = 0.0
        self.last_input = None

    def calculate_output(self, inputs):
        inputs_with_bias = np.append(inputs, 1.0)
        self.last_input = inputs_with_bias
        weighted_sum = np.dot(inputs_with_bias, self.weights)
        self.output = self.relu(weighted_sum)
        return self.output

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = np.array(weights)
        
    def relu(self, x):
        return np.maximum(0, x)

    def __repr__(self):
        return f"Neuron(weights={self.weights})"