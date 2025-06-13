from .neuron import Neuron
from array import array

class Layer:
    def __init__(self, numNeurons, numWeights) -> None:
        self.numWeights = numWeights
        self.numNeurons = numNeurons
        self.neurons = [Neuron(numWeights) for _ in range(numNeurons)]

    def update_weights(self, weights):
        if weights.shape != (self.numNeurons, self.numWeights):
            raise ValueError(f"Expected weights shape ({self.numNeurons}, {self.numWeights}), got {weights.shape}")
            
        for i in range(self.numNeurons):
            self.neurons[i].update_weights(weights[i])

    def calculate_layer_output(self, entrys) -> array:
        outputs = array('f', [0.0] * self.numNeurons)

        for i in range(self.numNeurons):
            outputs[i] = float(self.neurons[i].calculate_output(entrys))

        return outputs
    
    def __str__(self) -> str:
        return '\n'.join(f'Neuron {i}: {self.neurons[i]}' for i in range(len(self.neurons)))
