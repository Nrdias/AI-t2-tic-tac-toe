import numpy as np


class NeuralNetwork:
    """Implementa uma Rede Neural MLP com propagação direta."""

    def __init__(self, topology):
        self.topology = topology
        self.weights = []
        self.biases = []
        for i in range(len(topology) - 1):
            weight_matrix = np.random.randn(topology[i + 1], topology[i])
            bias_vector = np.random.randn(topology[i + 1], 1)
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def relu(self, x):
        return np.maximum(0, x)

    def set_weights_from_chromosome(self, chromosome):
        pointer = 0
        new_weights = []
        new_biases = []
        for i in range(len(self.topology) - 1):
            rows, cols = self.topology[i + 1], self.topology[i]
            size = rows * cols
            layer_weights = np.array(chromosome[pointer : pointer + size]).reshape(
                rows, cols
            )
            new_weights.append(layer_weights)
            pointer += size

            size = rows
            layer_biases = np.array(chromosome[pointer : pointer + size]).reshape(
                rows, 1
            )
            new_biases.append(layer_biases)
            pointer += size

        self.weights = new_weights
        self.biases = new_biases

    def propagate(self, inputs):
        activations = np.array(inputs).reshape(self.topology[0], 1)
        for weights, biases in zip(self.weights, self.biases):
            z = np.dot(weights, activations) + biases
            activations = self.relu(z)
        return activations.flatten()

    def choose_move(self, game):
        nn_input = game.get_board_for_nn()
        output = self.propagate(nn_input)

        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        move_preferences = sorted(enumerate(output), key=lambda x: x[1], reverse=True)

        for move_index, _ in move_preferences:
            move = (move_index // 3, move_index % 3)
            if move in valid_moves:
                return move
        return None
