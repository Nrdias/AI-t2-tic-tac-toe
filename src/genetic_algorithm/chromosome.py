import numpy as np


class Chromosome:
    def __init__(self, weights):
        self.weights = np.array(weights, dtype=np.float64)
        self.fitness = 0.0

    def __len__(self):
        return len(self.weights)

    def __repr__(self):
        return (
            f"Chromosome(fitness={self.fitness:.2f}, weights_len={len(self.weights)})"
        )
