import numpy as np
from src.config import EMPTY_CELL, PLAYER_X, PLAYER_O


class Board:
    def __init__(self):
        self.board = np.full((3, 3), EMPTY_CELL, dtype=int)

    def get_state(self):
        """Retorna o estado atual do tabuleiro como um vetor unidimensional."""
        return self.board.flatten()

    def make_move(self, row, col, player):
        """Tenta fazer uma jogada no tabuleiro."""
        if 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == EMPTY_CELL:
            self.board[row, col] = player
            return True
        return False

    def is_full(self):
        """Verifica se o tabuleiro está cheio (empate)."""
        return EMPTY_CELL not in self.board

    def check_win(self, player):
        """Verifica se o jogador venceu."""
        for i in range(3):
            if np.all(self.board[i, :] == player):
                return True
            if np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player):
            return True
        if np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def reset(self):
        """Reseta o tabuleiro para o estado inicial."""
        self.board = np.full((3, 3), EMPTY_CELL, dtype=int)

    def display(self):
        """Exibe o tabuleiro no console (para debug)."""
        symbol_map = {PLAYER_X: "X", PLAYER_O: "O", EMPTY_CELL: "-"}
        for row in self.board:
            print(" ".join([symbol_map[cell] for cell in row]))
        print("-" * 5)
