import random
import time

import numpy as np

from src.config import FITNESS_WEIGHTS, NN_PLAYER_SYMBOL, OPPONENT_PLAYER_SYMBOL
from src.neural_network import NeuralNetwork
from src.board import Board


class GeneticAlgorithm:
    """Gerencia a evolução da população de redes neurais."""

    def __init__(
        self,
        population_size,
        chromosome_length,
        mutation_rate,
        crossover_rate,
        elitism_count,
    ):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.population = self._create_initial_population()

    def _create_initial_population(self):
        """Cria a população inicial com pesos de cromossomos aleatórios."""
        pop = []
        for _ in range(self.population_size):
            chromosome = np.random.uniform(-1.0, 1.0, self.chromosome_length).tolist()
            pop.append(chromosome)
        return pop

    def _calculate_fitness(self, chromosome, minimax_player, visualize=False):
        """
        Calcula a aptidão de um cromossomo jogando contra o Minimax,
        usando um sistema de tags para pontuar as jogadas.
        """
        nn = NeuralNetwork(topology=[9, 9, 9])
        nn.set_weights_from_chromosome(chromosome)
        board = Board()

        move_tags = []

        # Resultado padrão caso o jogo termine por jogada inválida
        result = "invalid"

        if visualize:
            print("\n--- Novo Jogo de Treinamento ---")

        while not board.is_game_over():
            # Vez da Rede Neural
            move = nn.choose_move(board)

            if not board.is_move_valid(move):
                try:
                    symbol_at_pos = board.board[move[0]][move[1]]
                    if symbol_at_pos == NN_PLAYER_SYMBOL:
                        move_tags.append("invalid_own_pos")
                    else:
                        move_tags.append("invalid_other_pos")
                except (TypeError, IndexError):
                    move_tags.append("invalid_other_pos")
                break

            if self._is_winning_move(board, move, NN_PLAYER_SYMBOL):
                move_tags.append("victory")
            if self._is_defensive_move(board, move, OPPONENT_PLAYER_SYMBOL):
                move_tags.append("defensive")
            board.make_move(move, NN_PLAYER_SYMBOL)
            if "victory" not in move_tags and "defensive" not in move_tags:
                move_tags.append("offensive")
            board.undo_move(move)

            if self._is_priority_move(move):
                move_tags.append("prioritary")

            board.make_move(move, NN_PLAYER_SYMBOL)
            move_tags.append("in_progress")  # Recompensa por fazer uma jogada válida

            if visualize:
                print("--- Jogada da Rede Neural ---")
                board.print_board()
                time.sleep(0.3)

            if board.is_game_over():
                break

            # Vez do Minimax
            minimax_move = minimax_player.choose_move(board, OPPONENT_PLAYER_SYMBOL)
            board.make_move(minimax_move, OPPONENT_PLAYER_SYMBOL)

            if visualize:
                print("--- Jogada do Minimax ---")
                board.print_board()
                time.sleep(0.3)

        if board.current_winner == NN_PLAYER_SYMBOL:
            result = "win"
        elif board.current_winner == OPPONENT_PLAYER_SYMBOL:
            move_tags.append("defeat")
            result = "loss"
        elif board.is_draw():
            move_tags.append("draw")
            result = "draw"

        fitness = sum(FITNESS_WEIGHTS.get(tag, 0) for tag in move_tags)

        if visualize:
            print(f"Tags da Partida: {move_tags}")
            print(f"Resultado: {result.upper()} | Aptidão Final: {fitness:.2f}")
            time.sleep(0.5)

        return fitness, result

    def _is_winning_move(self, board, move, symbol):
        """Verifica se uma jogada resulta em vitória."""
        board.make_move(move, symbol)
        is_winner = board.check_winner(symbol)
        board.undo_move(move)
        return is_winner

    def _is_defensive_move(self, board, move, opponent_symbol):
        """Verifica se a jogada impede uma vitória iminente do oponente."""

        for r, c in board.get_valid_moves():
            if (r, c) != move:
                if self._is_winning_move(board, (r, c), opponent_symbol):
                    if move == (r, c):
                        return True
        return False

    def _is_priority_move(self, move):
        """Verifica se a jogada é em uma posição estratégica (centro ou cantos)."""
        center = (1, 1)
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        return move == center or move in corners

    def _selection(self, pop_with_fitness):
        """Seleção por torneio para escolher os pais."""
        tournament_size = 5
        selected = []
        for _ in range(2):
            participants = random.sample(pop_with_fitness, tournament_size)
            winner = max(participants, key=lambda x: x[1])
            selected.append(winner[0])
        return selected[0], selected[1]

    def _crossover(self, parent1, parent2):
        """Crossover aritmético para gerar filhos."""
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]  # Retorna cópias

        child1, child2 = list(parent1), list(parent2)
        alpha = random.random()
        for i in range(self.chromosome_length):
            child1[i] = parent1[i] * alpha + parent2[i] * (1 - alpha)
            child2[i] = parent2[i] * alpha + parent1[i] * (1 - alpha)
        return child1, child2

    def _mutate(self, chromosome):
        """Mutação gaussiana para adicionar variabilidade."""
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                chromosome[i] += np.random.normal(0, 0.2)
        return chromosome

    def evolve(self, minimax_player, slow_mode=False):
        """Executa um ciclo de evolução da população (uma geração)."""

        fitness_results = [
            self._calculate_fitness(chrom, minimax_player, slow_mode)
            for chrom in self.population
        ]

        pop_with_fitness = [
            (self.population[i], fitness_results[i][0])
            for i in range(len(self.population))
        ]
        game_outcomes = [res[1] for res in fitness_results]

        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)

        all_fitness_scores = [individual[1] for individual in pop_with_fitness]
        best_fitness = all_fitness_scores[0]
        worst_fitness = all_fitness_scores[-1]
        average_fitness = sum(all_fitness_scores) / len(all_fitness_scores)

        best_chromosome = pop_with_fitness[0][0]
        chromosome_snippet = ", ".join(
            [f"{weight:.3f}" for weight in best_chromosome[:6]]
        )

        wins = game_outcomes.count("win")
        draws = game_outcomes.count("draw")
        losses = game_outcomes.count("loss")
        total_games = len(game_outcomes)
        accuracy = ((wins + draws) / total_games) * 100 if total_games > 0 else 0

        print(
            f"Aptidão (Melhor/Média/Pior): {best_fitness:>6.2f} / {average_fitness:>6.2f} / {worst_fitness:>6.2f} | "
            f"Acurácia: {accuracy:.2f}% | "
            f"V/E/D: {wins}/{draws}/{losses}"
        )
        print(f"  └ Melhor Cromossomo (início): [{chromosome_snippet}, ...]")

        next_generation = [best_chromosome]  # Elitismo: mantém o melhor cromossomo

        while len(next_generation) < self.population_size:
            parent1, parent2 = self._selection(pop_with_fitness)
            child1, child2 = self._crossover(parent1, parent2)
            next_generation.append(self._mutate(child1))
            if len(next_generation) < self.population_size:
                next_generation.append(self._mutate(child2))

        self.population = next_generation
        return pop_with_fitness[0]
