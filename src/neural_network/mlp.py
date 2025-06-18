import numpy as np
from src.neural_network.neuron import Neuron

class MLP:
    def __init__(self, input_neurons, hidden_layer_neurons, output_neurons):
        self.input_neurons_count = input_neurons
        self.hidden_layer_neurons_count = hidden_layer_neurons
        self.output_neurons_count = output_neurons

        self.layers = []

        hidden_layer = [Neuron(input_neurons) for _ in range(hidden_layer_neurons)]
        self.layers.append(hidden_layer)

        output_layer = [Neuron(hidden_layer_neurons) for _ in range(output_neurons)]
        self.layers.append(output_layer)

    def forward_propagation(self, inputs):
        current_inputs = np.array(inputs)
        for layer in self.layers:
            next_inputs = []
            for neuron in layer:
                next_inputs.append(neuron.calculate_output(current_inputs))
            current_inputs = np.array(next_inputs)
        return current_inputs

    def get_all_weights(self):
        all_weights = []
        for layer in self.layers:
            for neuron in layer:
                all_weights.extend(neuron.get_weights())
        return np.array(all_weights)

    def set_all_weights(self, weights):
        weight_idx = 0
        for layer in self.layers:
            for neuron in layer:
                num_weights_neuron = len(neuron.get_weights())
                neuron_weights = weights[weight_idx : weight_idx + num_weights_neuron]
                neuron.set_weights(neuron_weights)
                weight_idx += num_weights_neuron

    def get_total_weights_count(self):
        count = 0
        count += (self.input_neurons_count + 1) * self.hidden_layer_neurons_count
        count += (self.hidden_layer_neurons_count + 1) * self.output_neurons_count
        return count