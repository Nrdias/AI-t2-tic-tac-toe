import random
import numpy as np

from src.game_environment.tictactoe_game import TicTacToeGame
from src.genetic_algorithm.chromosome import Chromosome
from src.genetic_algorithm.operators import crossover, mutation, selection
from src.minimax.minimax_player import MinimaxPlayer
from src.neural_network.mlp import MLP
from src.config import (
    DRAW_SCORE,
    EMPTY_CELL,
    HIDDEN_LAYER_1_NEURONS,
    INPUT_NEURONS,
    INVALID_MOVE_PENALTY,
    LOSS_PENALTY,
    OUTPUT_NEURONS,
    PLAYER_O,
    PLAYER_X,
    TRAINING_MODES_ORDER,
    WIN_SCORE,
)


class GeneticAlgorithm:
    def __init__(
        self, population_size, generations, mutation_rate, crossover_rate, elitism_count
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.population = []
        self.best_chromosome = None
        self.fitness_history = []

        temp_mlp = MLP(INPUT_NEURONS, HIDDEN_LAYER_1_NEURONS, OUTPUT_NEURONS)
        self.chromosome_length = temp_mlp.get_total_weights_count()
        del temp_mlp

    def initialize_population(self):
        print("Inicializando população...")
        for _ in range(self.population_size):
            weights = np.random.randn(self.chromosome_length) * 0.1
            self.population.append(Chromosome(weights))
        print(f"População inicial de {self.population_size} cromossomos criada.")

    def evolve(self):
        print("Iniciando evolução do Algoritmo Genético...")
        current_training_mode_idx = 0

        for generation in range(self.generations):
            if len(TRAINING_MODES_ORDER) > 0:
                mode_switch_interval = max(
                    1, self.generations // len(TRAINING_MODES_ORDER)
                )

                if (
                    generation % mode_switch_interval == 0
                    and current_training_mode_idx < len(TRAINING_MODES_ORDER)
                ):
                    minimax_difficulty = TRAINING_MODES_ORDER[current_training_mode_idx]
                    current_training_mode_idx += 1
                elif current_training_mode_idx >= len(TRAINING_MODES_ORDER):
                    minimax_difficulty = TRAINING_MODES_ORDER[-1]
                else:
                    minimax_difficulty = TRAINING_MODES_ORDER[
                        current_training_mode_idx - 1
                    ]
            else:
                minimax_difficulty = "hard"

            print(
                f"\n--- Geração {generation + 1}/{self.generations} (Modo Minimax: {minimax_difficulty.upper()}) ---"
            )

            # Calcular aptidão de cada cromossomo
            for i, chromosome in enumerate(self.population):
                chromosome.fitness = self.calculate_fitness(
                    chromosome.weights, minimax_difficulty
                )

            current_best_chromosome = max(
                self.population, key=lambda chrom: chrom.fitness
            )
            if (
                self.best_chromosome is None
                or current_best_chromosome.fitness > self.best_chromosome.fitness
            ):
                self.best_chromosome = current_best_chromosome

            avg_fitness = np.mean([c.fitness for c in self.population])
            self.fitness_history.append(avg_fitness)

            print(
                f"  Aptidão Média: {avg_fitness:.2f}, Melhor Aptidão (atual): {current_best_chromosome.fitness:.2f}, Melhor Aptidão (geral): {self.best_chromosome.fitness:.2f}"
            )

            if (
                len(self.fitness_history) > 10
                and (
                    np.max(self.fitness_history[-10:])
                    - np.min(self.fitness_history[-10:])
                )
                < 1.0
            ):
                print("Convergência detectada. Parando evolução.")
                break

            selected_population = selection(self.population, self.elitism_count)
            new_population = selected_population[:]

            while len(new_population) < self.population_size:
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)

                child1_weights, child2_weights = crossover(
                    parent1, parent2, self.crossover_rate
                )

                mutated_child1_weights = mutation(child1_weights, self.mutation_rate)
                mutated_child2_weights = mutation(child2_weights, self.mutation_rate)

                new_population.append(Chromosome(mutated_child1_weights))
                if len(new_population) < self.population_size:
                    new_population.append(Chromosome(mutated_child2_weights))

            self.population = new_population[: self.population_size]

        print("\nEvolução do Algoritmo Genético finalizada.")
        print(
            f"Melhor Cromossomo encontrado (final): Aptidão = {self.best_chromosome.fitness:.2f}"
        )
        return self.best_chromosome, self.fitness_history

    def calculate_fitness(self, chromosome_weights, minimax_difficulty):
        fitness = 0.0

        nn_player = MLP(INPUT_NEURONS, HIDDEN_LAYER_1_NEURONS, OUTPUT_NEURONS)
        nn_player.set_all_weights(chromosome_weights)

        def nn_make_move_decision_for_game(board_state_flat):
            outputs = nn_player.forward_propagation(board_state_flat)
            sorted_indices = np.argsort(outputs)[::-1]

            for idx in sorted_indices:
                r, c = divmod(idx, 3)
                if board_state_flat[idx] == EMPTY_CELL:
                    return r, c
            return -1, -1

        nn_player.make_move_decision = nn_make_move_decision_for_game

        minimax_opponent = MinimaxPlayer(
            PLAYER_O, PLAYER_X, difficulty_mode=minimax_difficulty
        )
        game = TicTacToeGame(player1=nn_player, player2=minimax_opponent)

        game_over = False

        while not game_over:
            current_player_at_turn_start = game.current_player

            is_valid_move = game.play_turn()

            winner, game_over = game.get_game_result()

            if not is_valid_move and current_player_at_turn_start == PLAYER_X:
                fitness += INVALID_MOVE_PENALTY
                game_over = True
                break

            if game_over:
                if winner == PLAYER_X:
                    fitness += WIN_SCORE
                elif winner == PLAYER_O:
                    fitness += LOSS_PENALTY
                elif winner == EMPTY_CELL:
                    fitness += DRAW_SCORE
                break

        return fitness
