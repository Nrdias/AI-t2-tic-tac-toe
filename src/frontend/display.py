import matplotlib.pyplot as plt

def plot_fitness_history(fitness_history):
    """
    Plota o histórico da aptidão média da população ao longo das gerações.
    """
    if not fitness_history:
        print("Nenhum histórico de aptidão para plotar.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, marker='o', linestyle='-', color='b')
    plt.title('Evolução da Aptidão Média da População')
    plt.xlabel('Geração')
    plt.ylabel('Aptidão Média')
    plt.grid(True)
    plt.show()

def print_evolution_step(generation, avg_fitness, max_fitness, best_chromosome_fitness):
    """
    Imprime um passo da evolução para acompanhar o AG. 
    """
    print(f"Geração {generation}:")
    print(f"  Aptidão Média: {avg_fitness:.2f}")
    print(f"  Melhor Aptidão (nesta geração): {max_fitness:.2f}")
    print(f"  Melhor Aptidão (geral até agora): {best_chromosome_fitness:.2f}")
    print("-" * 30)