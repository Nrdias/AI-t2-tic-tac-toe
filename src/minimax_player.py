import math
import random

from src.config import NN_PLAYER_SYMBOL, OPPONENT_PLAYER_SYMBOL


class MinimaxPlayer:
    """Implementa o jogador Minimax que servirá de treinador."""

    def __init__(self, difficulty="hard"):
        self.difficulty = difficulty

    def choose_move(self, game, player_symbol):
        """Escolhe um movimento baseado no nível de dificuldade."""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        if self.difficulty == "hard":
            return self.minimax(game, player_symbol)["position"]

        if self.difficulty == "medium":
            if random.random() < 0.50:
                return self.minimax(game, player_symbol)["position"]
            else:
                return random.choice(valid_moves)

    # Algoritmo Minimax otimizado para não usar deepcopy
    def minimax(self, state_game, player_symbol):
        """Algoritmo Minimax otimizado para encontrar o melhor movimento."""
        max_player = NN_PLAYER_SYMBOL
        other_player = (
            OPPONENT_PLAYER_SYMBOL if player_symbol == max_player else NN_PLAYER_SYMBOL
        )

        if state_game.current_winner == other_player:
            score = len(state_game.get_valid_moves()) + 1
            if other_player == max_player:
                return {"position": None, "score": score}
            else:
                return {"position": None, "score": -score}
        elif not state_game.get_valid_moves():
            return {"position": None, "score": 0}

        if player_symbol == max_player:
            best = {"position": None, "score": -math.inf}
        else:
            best = {"position": None, "score": math.inf}

        for possible_move in state_game.get_valid_moves():
            # Faz o movimento
            state_game.make_move(possible_move, player_symbol)

            # Chamada recursiva
            sim_score = self.minimax(state_game, other_player)

            # Desfaz o movimento
            state_game.undo_move(possible_move)

            sim_score["position"] = possible_move

            if player_symbol == max_player:
                if sim_score["score"] > best["score"]:
                    best = sim_score
            else:
                if sim_score["score"] < best["score"]:
                    best = sim_score
        return best
