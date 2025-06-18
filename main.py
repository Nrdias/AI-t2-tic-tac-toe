from src.genetic_algorithm.ga import (
    GeneticAlgorithm,
)
from src.frontend.console_interface import ConsoleInterface
from src.frontend.display import plot_fitness_history

from src.config import (
    POPULATION_SIZE,
    GENERATIONS,
    MUTATION_RATE,
    CROSSOVER_RATE,
    ELITISM_COUNT,
)


def run_training_mode():
    """
    Executa o processo de aprendizado da rede neural com o Algoritmo Genético.
    """
    print("--- INICIANDO TREINAMENTO DA REDE NEURAL ---")
    ga = GeneticAlgorithm(
        POPULATION_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, ELITISM_COUNT
    )
    ga.initialize_population()

    final_best_chromosome, fitness_history = ga.evolve()

    print("\n--- TREINAMENTO CONCLUÍDO ---")
    print(
        f"Pesos da melhor rede neural treinada (primeiros 10): {final_best_chromosome.weights[:10]}..."
    )

    plot_fitness_history(fitness_history)

    return final_best_chromosome.weights


def main():
    trained_weights = None

    while True:
        print("\n--- MENU PRINCIPAL ---")
        print("1. Treinar a Rede Neural")
        print("2. Jogar com o Minimax (Difícil)")
        print("3. Jogar com a Rede Neural Treinada")
        print("4. Calcular Acurácia da Rede Treinada")
        print("5. Sair")
        print("----------------------")

        choice = input("Digite sua escolha: ").strip()

        if choice == "1":
            trained_weights = run_training_mode()
            print(
                "\nTreinamento finalizado. A rede neural está pronta para ser testada."
            )
        elif choice == "2":
            console_game = ConsoleInterface(trained_nn_weights=None)
            console_game.play_game(mode="user_vs_minimax")
        elif choice == "3":
            if trained_weights is None:
                print(
                    "Aviso: Rede neural não foi treinada ainda. Por favor, treine primeiro (Opção 1)."
                )
                continue
            # Ao criar ConsoleInterface, ela importará suas próprias dependências,
            # mas agora MinimaxPlayer e MLP estão disponíveis no ga_core (ou no main.py se importados).
            # É importante que ConsoleInterface consiga encontrar MLP e MinimaxPlayer.
            # Como eles foram consolidados em ga_core, e main.py importa ga_core,
            # as classes estarão acessíveis via ga_core.MLP, ga_core.MinimaxPlayer etc.
            # Ajuste em ConsoleInterface para usar ga_core.MLP e ga_core.MinimaxPlayer
            # ou remover essas classes de ConsoleInterface e deixá-la puramente como uma interface.
            # Para evitar mais complexidade, a ConsoleInterface também precisa ser refatorada para
            # não usar imports 'src.' para MLP, MinimaxPlayer etc.
            # Isso significa que ConsoleInterface.py também precisará ter suas dependências
            # movidas para dentro dela, ou importá-las do ga_core.py.

            # Para o momento, assumindo que ConsoleInterface.py TAMBÉM será atualizado
            # para não usar src.* e, em vez disso, referenciar classes como MLP e MinimaxPlayer
            # que poderiam ser importadas do ga_core.py ou ter suas definições duplicadas para ela.
            console_game = ConsoleInterface(trained_nn_weights=trained_weights)
            console_game.play_game(mode="user_vs_nn")
        elif choice == "4":
            if trained_weights is None:
                print(
                    "Aviso: Rede neural não foi treinada ainda. Por favor, treine primeiro (Opção 1)."
                )
                continue
            console_game = ConsoleInterface(trained_nn_weights=trained_weights)
            console_game.calculate_and_display_accuracy()
        elif choice == "5":
            print("Saindo...")
            break
        else:
            print("Escolha inválida. Por favor, tente novamente.")


if __name__ == "__main__":
    main()
