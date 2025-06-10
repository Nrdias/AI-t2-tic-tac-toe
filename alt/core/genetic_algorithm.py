import random
import numpy as np
from src.core.individual import Individual
from tic_tac_toe import JogoDaVelha
from minimax import jogada_minimax


class GA:
    def __init__(self, tamanho_pop=50):
        self.populacao = [Individual() for _ in range(tamanho_pop)]

    def evaluate(self, modo="facil"):
        for ind in self.populacao:
            rede = ind.get_network()
            pontos = 0
            for _ in range(5):
                jogo = JogoDaVelha()
                fim = None
                while fim is None:
                    jogada = rede.escolher_jogada(jogo.tabuleiro)
                    if jogada is None or not jogo.jogar(jogada):
                        pontos -= 1  # penaliza erro
                        break
                    fim = jogo.verificar_vencedor()
                    if fim is not None:
                        break
                    jogada_op = jogada_minimax(jogo.tabuleiro, modo)
                    jogo.jogar(jogada_op)
                    fim = jogo.verificar_vencedor()
                if fim == 1:
                    pontos += 1
                elif fim == 0:
                    pontos += 0.5
            ind.aptidao = pontos

    def evolve(self):
        # placeholder: seleção + cruzamento + mutação
        self.populacao.sort(key=lambda x: x.aptidao, reverse=True)
        nova_geracao = self.populacao[:10]  # elitismo
        while len(nova_geracao) < len(self.populacao):
            pai, mae = random.sample(self.populacao[:20], 2)
            ponto = random.randint(1, len(pai.pesos) - 1)
            filho_pesos = np.concatenate((pai.pesos[:ponto], mae.pesos[ponto:]))
            filho_pesos += np.random.normal(0, 0.1, size=filho_pesos.shape)
            nova_geracao.append(Individual(filho_pesos))
        self.populacao = nova_geracao
