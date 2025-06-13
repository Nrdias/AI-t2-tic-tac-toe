import numpy as np
from .layer import Layer

class MLP:
    def __init__(self, weights=None):
        # Create layers with 9 neurons each
        self.hidden_layer = Layer(9, 10)  # 9 neurons, 10 weights each (including bias)
        self.output_layer = Layer(9, 10)  # 9 neurons, 10 weights each (including bias)
        
        if weights is not None:
            # Unpack weights from flat array
            weights = np.array(weights)
            if len(weights) != 180:  # 90 weights for hidden layer + 90 for output layer
                raise ValueError(f"Expected 180 weights, got {len(weights)}")
                
            # First 90 weights for hidden layer
            hidden_weights = weights[:90].reshape(9, 10)
            # Next 90 weights for output layer
            output_weights = weights[90:180].reshape(9, 10)
            
            # Update layer weights
            self.hidden_layer.update_weights(hidden_weights)
            self.output_layer.update_weights(output_weights)

    def feedforward(self, entrada):
        entrada = np.array(entrada)
        # Hidden layer with tanh activation
        hidden_output = self.hidden_layer.calculate_layer_output(entrada)
        # Output layer with linear activation
        output = self.output_layer.calculate_layer_output(hidden_output)
        return output

    def choose_move(self, tabuleiro):
        saida = self.feedforward(tabuleiro)
        livres = [i for i in range(9) if tabuleiro[i] == 0]
        if not livres:
            return None
        # Mask invalid moves with negative infinity
        scores = np.array(saida)
        scores[~np.isin(np.arange(9), livres)] = float('-inf')
        return np.argmax(scores)

    def get_weights(self):
        """Returns all weights as a flat array"""
        # Get weights from both layers
        hidden_weights = np.array([neuron.weights for neuron in self.hidden_layer.neurons])
        output_weights = np.array([neuron.weights for neuron in self.output_layer.neurons])
        
        # Flatten and concatenate
        return np.concatenate([
            hidden_weights.flatten(),
            output_weights.flatten()
        ]).tolist()
