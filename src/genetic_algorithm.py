import random
import time

import numpy as np
from src.config import NN_PLAYER_SYMBOL, OPPONENT_PLAYER_SYMBOL
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
        pop = []
        for _ in range(self.population_size):
            chromosome = np.random.uniform(-1.0, 1.0, self.chromosome_length).tolist()
            pop.append(chromosome)
        return pop

    def _calculate_fitness(self, chromosome, minimax_player, visualize=False):
        """Calcula a aptidão e retorna o resultado do jogo."""
        nn = NeuralNetwork(topology=[9, 9, 9])
        nn.set_weights_from_chromosome(chromosome)
        board = Board()

        if visualize:
            print("\n--- Novo Jogo de Treinamento ---")

        fitness = 0
        current_player = "NN"

        while not board.is_game_over():
            if current_player == "NN":
                move = nn.choose_move(board)
                if board.make_move(move, NN_PLAYER_SYMBOL):
                    fitness += 1
                else:
                    fitness -= 200
                    return fitness, "invalid"
                current_player = "MINIMAX"
            else:
                move = minimax_player.choose_move(board, OPPONENT_PLAYER_SYMBOL)
                board.make_move(move, OPPONENT_PLAYER_SYMBOL)
                current_player = "NN"

            if visualize:
                board.print_board()
                time.sleep(0.3)

        result = "draw"
        if board.current_winner == NN_PLAYER_SYMBOL:
            fitness += 50
            result = "win"
        elif board.current_winner == OPPONENT_PLAYER_SYMBOL:
            fitness -= 100
            result = "loss"
        else:
            fitness += 20

        if visualize:
            print(f"Resultado do Jogo: {result.upper()} | Aptidão Final: {fitness}")
            time.sleep(0.5)

        return fitness, result

    def _selection(self, pop_with_fitness):
        tournament_size = 5
        selected = []
        for _ in range(2):
            participants = random.sample(pop_with_fitness, tournament_size)
            winner = max(participants, key=lambda x: x[1])
            selected.append(winner[0])
        return selected[0], selected[1]

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]  # Retorna cópias

        child1, child2 = list(parent1), list(parent2)
        alpha = random.random()
        for i in range(self.chromosome_length):
            child1[i] = parent1[i] * alpha + parent2[i] * (1 - alpha)
            child2[i] = parent2[i] * alpha + parent1[i] * (1 - alpha)
        return child1, child2

    def _mutate(self, chromosome):
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                chromosome[i] += np.random.normal(0, 0.2)  # desvio padrão da mutação
        return chromosome

    # `evolve` agora calcula e exibe acurácia e pode visualizar os jogos
    def evolve(self, minimax_player, slow_mode=False):
        """Executa um ciclo de evolução, calcula estatísticas, visualiza jogos e o melhor cromossomo."""

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

        # Cálculo de Aptidão Média e Pior Aptidão ---
        all_fitness_scores = [individual[1] for individual in pop_with_fitness]
        best_fitness = all_fitness_scores[0]
        worst_fitness = all_fitness_scores[-1]
        average_fitness = sum(all_fitness_scores) / len(all_fitness_scores)
        # ---------------------------------------------------------

        best_chromosome = pop_with_fitness[0][0]
        chromosome_snippet = ", ".join(
            [f"{weight:.3f}" for weight in best_chromosome[:6]]
        )

        wins = game_outcomes.count("win")
        draws = game_outcomes.count("draw")
        losses = game_outcomes.count("loss")
        total_games = len(game_outcomes)
        accuracy = ((wins + draws) / total_games) * 100 if total_games > 0 else 0

        # Saída de dados aprimorada com as novas estatísticas ---
        print(
            f"Aptidão (Melhor/Média/Pior): {best_fitness:>6.2f} / {average_fitness:>6.2f} / {worst_fitness:>6.2f} | "
            f"Acurácia: {accuracy:.2f}% | "
            f"V/E/D: {wins}/{draws}/{losses}"
        )
        print(f"  └ Melhor Cromossomo (início): [{chromosome_snippet}, ...]")

        next_generation = []

        elite = [item[0] for item in pop_with_fitness[: self.elitism_count]]
        next_generation.extend(elite)

        while len(next_generation) < self.population_size:
            parent1, parent2 = self._selection(pop_with_fitness)
            child1, child2 = self._crossover(parent1, parent2)
            next_generation.append(self._mutate(child1))
            if len(next_generation) < self.population_size:
                next_generation.append(self._mutate(child2))

        self.population = next_generation
        return pop_with_fitness[0]
