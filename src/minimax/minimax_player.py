import numpy as np
import random
from src.config import EMPTY_CELL, MINIMAX_MEDIUM_THRESHOLD


class MinimaxPlayer:
    def __init__(self, player_symbol, opponent_symbol, difficulty_mode="medium"):
        self.player_symbol = player_symbol
        self.opponent_symbol = opponent_symbol
        self.difficulty_mode = difficulty_mode

    def make_move_decision(self, board_state):
        current_board = np.array(board_state).reshape((3, 3))
        rand_val = random.random()

        if self.difficulty_mode == "medium" and rand_val > MINIMAX_MEDIUM_THRESHOLD:
            return self._get_random_move(current_board)

        best_score = -np.inf
        best_move = None

        available_moves = []
        for r in range(3):
            for c in range(3):
                if current_board[r, c] == EMPTY_CELL:
                    available_moves.append((r, c))

        if not available_moves:
            return None, None
        random.shuffle(available_moves)

        for r, c in available_moves:
            current_board[r, c] = self.player_symbol
            score = self._minimax(current_board, 0, False)
            current_board[r, c] = EMPTY_CELL
            if score > best_score:
                best_score = score
                best_move = (r, c)
        return best_move

    def _get_random_move(self, board):
        empty_cells = []
        for r in range(3):
            for c in range(3):
                if board[r, c] == EMPTY_CELL:
                    empty_cells.append((r, c))
        if empty_cells:
            return random.choice(empty_cells)
        return None, None

    def _minimax(self, board, depth, is_maximizing_player):
        if self._check_win_minimax(board, self.player_symbol):
            return 10 - depth
        if self._check_win_minimax(board, self.opponent_symbol):
            return -10 + depth
        if self._is_full_minimax(board):
            return 0

        if is_maximizing_player:
            best_score = -np.inf
            for r in range(3):
                for c in range(3):
                    if board[r, c] == EMPTY_CELL:
                        board[r, c] = self.player_symbol
                        score = self._minimax(board, depth + 1, False)
                        board[r, c] = EMPTY_CELL
                        best_score = max(best_score, score)
            return best_score
        else:
            best_score = np.inf
            for r in range(3):
                for c in range(3):
                    if board[r, c] == EMPTY_CELL:
                        board[r, c] = self.opponent_symbol
                        score = self._minimax(board, depth + 1, True)
                        board[r, c] = EMPTY_CELL
                        best_score = min(best_score, score)
            return best_score

    def _check_win_minimax(self, board, player):
        for i in range(3):
            if np.all(board[i, :] == player):
                return True
            if np.all(board[:, i] == player):
                return True
        if np.all(np.diag(board) == player):
            return True
        if np.all(np.diag(np.fliplr(board)) == player):
            return True
        return False

    def _is_full_minimax(self, board):
        return EMPTY_CELL not in board
