import numpy as np

class MLP:
    def __init__(self, pesos_input_oculta, pesos_oculta_saida):
        self.w1 = np.array(pesos_input_oculta).reshape((10, 9))  # oculta × entrada
        self.w2 = np.array(pesos_oculta_saida).reshape((9, 10))  # saída × oculta

    def activation(self, x):
        return np.maximum(0, x)

    def feedforward(self, entrada):
        entrada = np.array(entrada)
        oculta = self.activation(np.dot(self.w1, entrada))
        saida = np.dot(self.w2, oculta)
        return saida.tolist()

    def choose_move(self, tabuleiro):
        saida = self.feedforward(tabuleiro)
        livres = [i for i in range(9) if tabuleiro[i] == 0]
        jogadas_validas = {i: saida[i] for i in livres}
        return max(jogadas_validas, key=jogadas_validas.get) if jogadas_validas else None
