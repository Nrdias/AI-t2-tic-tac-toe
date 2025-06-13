from core.genetic_algorithm import GeneticAlgorithm
from core.tic_tac_toe import JogoDaVelha

def train():
    # Initialize genetic algorithm with parameters
    ga = GeneticAlgorithm()
    ga.taxa_de_crossover = 0.8
    ga.taxa_de_mutacao = 0.1
    ga.numero_maximo_geracoes = 100
    ga.tamanho_populacao = 30
    ga.elitismo = True
    
    # Run the genetic algorithm
    best_weights = ga.run_ga()
    
    print("\nTreinamento concluído!")
    print(f"Melhores pesos encontrados: {best_weights}")

if __name__ == "__main__":
    train()
