class JogoDaVelha:
    def __init__(self):
        self.tabuleiro = [0] * 9
        self.jogador_atual = 1

    def mostrar_tabuleiro(self):
        print([self.tabuleiro[i:i+3] for i in range(0, 9, 3)])

    def jogar(self, posicao):
        if self.tabuleiro[posicao] != 0:
            return False  # Jogada inválida
        self.tabuleiro[posicao] = self.jogador_atual
        self.jogador_atual *= -1
        return True

    def verificar_vencedor(self):
        linhas = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a,b,c in linhas:
            if self.tabuleiro[a] == self.tabuleiro[b] == self.tabuleiro[c] != 0:
                return self.tabuleiro[a]
        if 0 not in self.tabuleiro:
            return 0  # empate
        return None  # jogo continua
