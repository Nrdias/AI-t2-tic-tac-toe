import random
import math


class Minimax:
    def __init__(self, mode="medio"):
        self.mode = mode
        self.jogador = 'O'  # Jogador humano
        self.computador = 'X'  # Computador

    def verificar_vencedor(self, tabuleiro):
        """ Verifica se há um vencedor ou um empate """
        combinacoes = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Linhas
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Colunas
            (0, 4, 8), (2, 4, 6)              # Diagonais
        ]

        for comb in combinacoes:
            if tabuleiro[comb[0]] == tabuleiro[comb[1]] == tabuleiro[comb[2]] != 0:
                return tabuleiro[comb[0]]  # Retorna 'X' ou 'O' se houver um vencedor

        if 0 not in tabuleiro:
            return 'empate'  # Empate se não houver células vazias

        return None  # Jogo continua

    def minimax(self, tabuleiro, profundidade, is_maximizing):
        """ Algoritmo Minimax para calcular a jogada ideal """
        vencedor = self.verificar_vencedor(tabuleiro)

        if vencedor == self.jogador:  # Se o jogador humano vencer
            return -10 + profundidade
        elif vencedor == self.computador:  # Se o computador vencer
            return 10 - profundidade
        elif vencedor == 'empate':  # Se houver empate
            return 0

        if is_maximizing:  # Maximiza para o computador
            melhor_valor = -math.inf
            for i in range(9):
                if tabuleiro[i] == 0:  # Verifica se a posição está vazia
                    tabuleiro[i] = self.computador  # O computador faz a jogada
                    valor = self.minimax(tabuleiro, profundidade + 1, False)  # Recursão
                    tabuleiro[i] = 0  # Desfaz a jogada
                    melhor_valor = max(melhor_valor, valor)
            return melhor_valor
        else:  # Minimiza para o jogador humano
            melhor_valor = math.inf
            for i in range(9):
                if tabuleiro[i] == 0:  # Verifica se a posição está vazia
                    tabuleiro[i] = self.jogador  # O jogador faz a jogada
                    valor = self.minimax(tabuleiro, profundidade + 1, True)  # Recursão
                    tabuleiro[i] = 0  # Desfaz a jogada
                    melhor_valor = min(melhor_valor, valor)
            return melhor_valor

    def melhor_jogada(self, tabuleiro):
        """ Encontra a melhor jogada para o computador """
        melhor_valor = -math.inf
        melhor_movimento = -1

        for i in range(9):
            if tabuleiro[i] == 0:  # Verifica se a célula está vazia
                tabuleiro[i] = self.computador  # Computador faz a jogada
                valor = self.minimax(tabuleiro, 0, False)  # Simula o jogo
                tabuleiro[i] = 0  # Desfaz a jogada
                if valor > melhor_valor:
                    melhor_valor = valor
                    melhor_movimento = i  # Armazena o índice da melhor jogada

        return melhor_movimento

    def choose_move(self, tabuleiro) -> int | None:
        """Escolhe a próxima jogada baseada no modo de dificuldade"""
        # Verifica se há células livres
        livres = [i for i in range(9) if tabuleiro[i] == 0]
        if not livres:
            return None

        # Decide se vai usar minimax baseado no modo
        rand = random.random()
        usar_minimax = False

        if self.mode == "medio":
            usar_minimax = rand < 0.50  # 50% de chance de usar minimax
        else:  # modo dificil
            usar_minimax = True  # Sempre usa minimax

        if usar_minimax:
            return self.melhor_jogada(tabuleiro)
        else:
            return random.choice(livres)  # Jogada aleatória

    def move(self, tabuleiro) -> int | None:
        # TODO: sempre joga na primeira célula livre
        for i in range(9):
            if tabuleiro[i] == 0:
                return i
        return None
