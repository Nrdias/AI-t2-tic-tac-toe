import math
import numpy as np

class Neuron:
    def __init__(self, numWeights) -> None:
        self.numWeights = numWeights
        self.weights = [0.0] * numWeights
        self.activation_func = self.relu  # Default to relu
    
    def update_weights(self, weights) -> None: 
        if len(weights) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} weights, got {len(weights)}")
        for i in range(0, len(self.weights)):
            self.weights[i] = weights[i]

    def _adder(self, entrys) -> int:
        sum = 0
        for i in range(0, len(self.weights)):
            if i == 0:
                sum = sum + self.weights[0]
            else:
                sum = sum + self.weights[i] * entrys[i-1]

        return sum
        
    def calculate_output(self, entrys) -> float:
        sum = self._adder(entrys)
        return self.activation_func(sum)
        
    def relu(self, x):
        return max(0, x)

    def logistic(self, x):
        return 1 / (1 + math.exp(-x))

    def __str__(self) -> str:
        return '[' + ', '.join(f'{w:.2f}' for w in self.weights) + ']'