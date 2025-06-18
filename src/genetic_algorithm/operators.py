import numpy as np
import random


def selection(population, elitism_count):
    population.sort(key=lambda chrom: chrom.fitness, reverse=True)
    next_generation = population[:elitism_count]
    tournament_size = 5
    while len(next_generation) < len(population):
        tournament_participants = random.sample(population, tournament_size)
        winner = max(tournament_participants, key=lambda chrom: chrom.fitness)
        next_generation.append(winner)
    return next_generation


def crossover(parent1, parent2, crossover_rate):
    if random.random() > crossover_rate:
        return parent1.weights, parent2.weights

    alpha = 0.5
    child1_weights = np.zeros_like(parent1.weights)
    child2_weights = np.zeros_like(parent2.weights)

    for i in range(len(parent1.weights)):
        child1_weights[i] = (
            alpha * parent1.weights[i] + (1 - alpha) * parent2.weights[i]
        )
        child2_weights[i] = (
            alpha * parent2.weights[i] + (1 - alpha) * parent1.weights[i]
        )

    return child1_weights, child2_weights


def mutation(chromosome_weights, mutation_rate):
    mutated_weights = np.copy(chromosome_weights)
    for i in range(len(mutated_weights)):
        if random.random() < mutation_rate:
            mutated_weights[i] += np.random.normal(0, 0.1)
    return mutated_weights
