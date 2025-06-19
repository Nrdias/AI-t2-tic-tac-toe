from src.config import (
    EMPTY_CELL,
    EMPTY_VAL,
    NN_PLAYER_SYMBOL,
    NN_PLAYER_VAL,
    OPPONENT_PLAYER_SYMBOL,
    OPPONENT_PLAYER_VAL,
)


class Board:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reinicia o tabuleiro para um novo jogo."""
        self.board = [[EMPTY_CELL] * 3 for _ in range(3)]
        self.current_winner = None

    def print_board(self):
        """Imprime o tabuleiro no console."""
        print("\n   0   1   2")
        for i, row in enumerate(self.board):
            print(f"{i}  {' | '.join(row)}")
            if i < 2:
                print("  ---|---|---")
        print()

    def get_valid_moves(self):
        """Retorna uma lista de tuplas (linha, coluna) para movimentos válidos."""
        return [
            (r, c) for r in range(3) for c in range(3) if self.board[r][c] == EMPTY_CELL
        ]

    def is_move_valid(self, move):
        """Verifica se um movimento (linha, coluna) é válido."""
        if move is None or not (0 <= move[0] < 3 and 0 <= move[1] < 3):
            return False
        return self.board[move[0]][move[1]] == EMPTY_CELL

    def is_draw(self):
        """Verifica se o jogo terminou em empate."""
        return self.current_winner is None and not self.get_valid_moves()

    def make_move(self, move, symbol):
        """Realiza um movimento no tabuleiro se for válido."""
        if self.is_move_valid(move):
            self.board[move[0]][move[1]] = symbol
            if self.check_winner(symbol):
                self.current_winner = symbol
            return True
        return False

    # Método para otimizar o Minimax, evitando deepcopy
    def undo_move(self, move):
        """Desfaz um movimento no tabuleiro."""
        self.board[move[0]][move[1]] = EMPTY_CELL
        self.current_winner = None  # Resetar o vencedor ao desfazer

    def check_winner(self, symbol):
        """Verifica se o jogador com o símbolo fornecido venceu."""
        won = False
        for i in range(3):
            if all(self.board[i][j] == symbol for j in range(3)):
                won = True
            if all(self.board[j][i] == symbol for j in range(3)):
                won = True
        if all(self.board[i][i] == symbol for i in range(3)):
            won = True
        if all(self.board[i][2 - i] == symbol for i in range(3)):
            won = True
        return won

    def is_game_over(self):
        """Verifica se o jogo terminou (vitória ou empate)."""
        return self.current_winner is not None or not self.get_valid_moves()

    def get_board_for_nn(self):
        """Converte o tabuleiro para o formato de entrada da rede neural."""
        nn_board = []
        for row in self.board:
            for cell in row:
                if cell == NN_PLAYER_SYMBOL:
                    nn_board.append(NN_PLAYER_VAL)
                elif cell == OPPONENT_PLAYER_SYMBOL:
                    nn_board.append(OPPONENT_PLAYER_VAL)
                else:
                    nn_board.append(EMPTY_VAL)
        return nn_board
