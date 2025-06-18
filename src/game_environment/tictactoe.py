from src.game_environment.board import Board
from src.config import PLAYER_X, PLAYER_O, EMPTY_CELL


class TicTacToe:
    def __init__(self, player1, player2):
        self.board = Board()
        self.player1 = player1
        self.player2 = player2
        self.current_player = PLAYER_X
        self.game_over = False
        self.winner = None
        self.move_count = 0

    def play_turn(self):
        if self.game_over:
            return False
        current_board_state = self.board.get_state()
        row, col = -1, -1
        is_valid_move = False

        if self.current_player == PLAYER_X:
            player_obj = self.player1
            player_symbol = PLAYER_X
        else:
            player_obj = self.player2
            player_symbol = PLAYER_O

        if player_obj == "Human":
            return False  # Lógica externa lida com jogador humano
        elif hasattr(player_obj, "make_move_decision"):
            row, col = player_obj.make_move_decision(current_board_state)
            if (
                row is not None
                and col is not None
                and self.board.board[row, col] == EMPTY_CELL
            ):
                is_valid_move = self.board.make_move(row, col, player_symbol)
            else:
                is_valid_move = False

        if is_valid_move:
            self.move_count += 1
            self._check_game_status()
            if not self.game_over:
                self._switch_player()
            return True
        else:
            return False

    def _check_game_status(self):
        if self.board.check_win(PLAYER_X):
            self.game_over = True
            self.winner = PLAYER_X
        elif self.board.check_win(PLAYER_O):
            self.game_over = True
            self.winner = PLAYER_O
        elif self.board.is_full():
            self.game_over = True
            self.winner = EMPTY_CELL

    def _switch_player(self):
        self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X

    def get_game_result(self):
        return self.winner, self.game_over

    def reset_game(self):
        self.board.reset()
        self.current_player = PLAYER_X
        self.game_over = False
        self.winner = None
        self.move_count = 0
