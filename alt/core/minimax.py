import random


class Minimax:
    def __init__(self, mode="facil"):
        self.mode = mode

    def choose_move(self, tabuleiro) -> int | None:
        rand = random.random()
        if self.mode == "facil":
            if rand < 0.50:
                return self.move(tabuleiro)
        elif self.mode == "medio":
            return self.move(tabuleiro)

        # TODO: jogada aleatória
        livres = [i for i in range(9) if tabuleiro[i] == 0]
        return random.choice(livres) if livres else None

    def move(self, tabuleiro) -> int | None:
        # TODO: sempre joga na primeira célula livre
        for i in range(9):
            if tabuleiro[i] == 0:
                return i
        return None
