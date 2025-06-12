import numpy as np
from .mlp import MLP


class Individual:
    def __init__(self, pesos=None):
        if pesos is None:
            self.pesos = np.random.uniform(-1, 1, size=(10 * 9 + 9 * 10))
        else:
            self.pesos = pesos
        self.aptidao = 0

    def get_network(self):
        w1 = self.pesos[:90]
        w2 = self.pesos[90:]
        return MLP(w1, w2)
