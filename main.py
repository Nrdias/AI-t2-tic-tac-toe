import csv
import math
import os
import time

from src.board import Board
from src.genetic_algorithm import GeneticAlgorithm
from src.minimax_player import MinimaxPlayer
from src.neural_network import NeuralNetwork
from src.config import (
    DEFAULT_GENS_HARD,
    DEFAULT_GENS_MEDIUM,
    DEFAULT_POP_SIZE,
    NN_PLAYER_SYMBOL,
    OPPONENT_PLAYER_SYMBOL,
    WEIGHTS_FILE,
)


def play_user_vs_minimax():
    """Modo de jogo: Usuário contra Minimax."""
    board = Board()
    minimax_player = MinimaxPlayer(difficulty="hard")

    print("\n--- Você vs. Minimax (Hard) ---")
    print("Você joga como 'X'. Escolha sua jogada com linha,coluna (ex: 1,2).")

    current_player = "USER"
    while not board.is_game_over():
        board.print_board()
        if current_player == "USER":
            try:
                row, col = map(
                    int, input("Sua vez. Digite (linha,coluna): ").split(",")
                )
                if not board.make_move((row, col), NN_PLAYER_SYMBOL):
                    print("Movimento inválido. Tente novamente.")
                    continue
                current_player = "MINIMAX"
            except (ValueError, IndexError):
                print("Entrada inválida. Use o formato 'linha,coluna'.")
                continue
        else:  # Vez do Minimax
            print("Vez do Minimax...")
            time.sleep(0.5)
            move = minimax_player.choose_move(board, OPPONENT_PLAYER_SYMBOL)
            board.make_move(move, OPPONENT_PLAYER_SYMBOL)
            current_player = "USER"

    board.print_board()
    if board.current_winner:
        winner_name = "Você" if board.current_winner == NN_PLAYER_SYMBOL else "Minimax"
        print(f"Fim de jogo! Vencedor: {winner_name}")
    else:
        print("Fim de jogo! Empate.")


def train_nn_vs_minimax(
    population_size, generations_medium, generations_hard, slow_mode=False
):
    """Modo de treinamento: Rede Neural aprende com Minimax."""
    print("\n--- Treinando a Rede Neural com o Algoritmo Genético ---")

    topology = [9, 9, 9]
    chromosome_length = (topology[0] * topology[1] + topology[1]) + (
        topology[1] * topology[2] + topology[2]
    )

    ag = GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=chromosome_length,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elitism_count=2,
    )

    minimax_trainer = MinimaxPlayer(difficulty="medium")
    best_chromosome_overall = (None, -math.inf)

    if generations_medium > 0:
        print(
            f"\nFASE 1: Treinando com Minimax 'Médio' por {generations_medium} gerações..."
        )
        for gen in range(generations_medium):
            print(f"Geração {gen + 1}/{generations_medium} | ", end="")
            best_of_gen = ag.evolve(minimax_trainer, slow_mode)
            if best_of_gen[1] > best_chromosome_overall[1]:
                best_chromosome_overall = best_of_gen

    if generations_hard > 0:
        print(
            f"\nFASE 2: Treinando com Minimax 'Difícil' por {generations_hard} gerações..."
        )
        minimax_trainer.difficulty = "hard"
        for gen in range(generations_hard):
            print(f"Geração {gen + 1}/{generations_hard} | ", end="")
            best_of_gen = ag.evolve(minimax_trainer, slow_mode)
            if best_of_gen[1] > best_chromosome_overall[1]:
                best_chromosome_overall = best_of_gen

    print("\n--- Treinamento Concluído! ---")
    if best_chromosome_overall[0] is not None:
        print(f"Melhor aptidão final encontrada: {best_chromosome_overall[1]:.2f}")
    return best_chromosome_overall[0]


def play_user_vs_trained_nn(best_weights):
    """Modo de jogo: Usuário contra a Rede Neural treinada."""
    if not best_weights:
        print(
            "\nA Rede Neural ainda não foi treinada. Por favor, execute a opção 2 primeiro."
        )
        return

    nn = NeuralNetwork(topology=[9, 9, 9])
    nn.set_weights_from_chromosome(best_weights)

    print("\n--- Você vs. Rede Neural Treinada ---")
    print("Você joga como 'O'. A Rede Neural ('X') começa.")

    scores = {"nn_wins": 0, "user_wins": 0, "draws": 0}

    while True:
        board = Board()
        current_player = "NN"

        while not board.is_game_over():
            board.print_board()
            if current_player == "NN":
                print("Vez da Rede Neural...")
                time.sleep(1)
                move = nn.choose_move(board)
                board.make_move(move, NN_PLAYER_SYMBOL)
                current_player = "USER"
            else:
                try:
                    row, col = map(
                        int,
                        input(
                            f"Sua vez ({OPPONENT_PLAYER_SYMBOL}). Digite (linha,coluna): "
                        ).split(","),
                    )
                    if not board.make_move((row, col), OPPONENT_PLAYER_SYMBOL):
                        print("Movimento inválido. Tente novamente.")
                        continue
                    current_player = "NN"
                except (ValueError, IndexError):
                    print("Entrada inválida. Use o formato 'linha,coluna'.")
                    continue

        board.print_board()
        if board.current_winner == NN_PLAYER_SYMBOL:
            print("Fim de jogo! A Rede Neural venceu.")
            scores["nn_wins"] += 1
        elif board.current_winner == OPPONENT_PLAYER_SYMBOL:
            print("Fim de jogo! Você venceu.")
            scores["user_wins"] += 1
        else:
            print("Fim de jogo! Empate.")
            scores["draws"] += 1

        total_games = sum(scores.values())
        accuracy = (
            ((scores["nn_wins"] + scores["draws"]) / total_games) * 100
            if total_games > 0
            else 0
        )
        print("\n--- Estatísticas da Partida ---")
        print(
            f"Vitórias da Rede: {scores['nn_wins']} | Suas Vitórias: {scores['user_wins']} | Empates: {scores['draws']}"
        )
        print(f"Acurácia da Rede (não perder): {accuracy:.2f}%")

        if input("\nJogar novamente? (s/n): ").lower() != "s":
            break


def save_weights_to_file(weights, filename=WEIGHTS_FILE):
    """Salva os pesos da rede neural em um arquivo CSV, um peso por linha."""
    if weights is None:
        print(
            "\n[AVISO] Nenhum peso para salvar, o treinamento pode não ter sido bem-sucedido."
        )
        return
    try:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            for weight in weights:
                writer.writerow([weight])
        print(f"\n[INFO] Melhores pesos salvos com sucesso em '{filename}'.")
    except Exception as e:
        print(f"\n[ERRO] Falha ao salvar os pesos: {e}")


def load_weights_from_file(filename=WEIGHTS_FILE):
    """Carrega os pesos de um arquivo CSV, detectando o formato automaticamente."""
    if not os.path.exists(filename):
        print(f"\n[AVISO] Arquivo de pesos '{filename}' não encontrado.")
        return None
    try:
        with open(filename, "r", newline="") as f:
            reader = csv.reader(f)
            first_row = next(reader)

            weights_str = []
            if len(first_row) > 1:
                print(
                    "\n[INFO] Formato antigo de pesos detectado. Carregando de uma única linha."
                )
                weights_str = first_row
            else:
                print("\n[INFO] Formato de um peso por linha detectado. Carregando...")
                weights_str.append(first_row[0])
                for row in reader:
                    if row:
                        weights_str.append(row[0])

            weights = [float(w) for w in weights_str]

        print(f"[INFO] {len(weights)} pesos carregados com sucesso de '{filename}'.")
        return weights
    except (StopIteration, ValueError):
        print(f"\n[ERRO] O arquivo de pesos '{filename}' está vazio ou corrompido.")
        return None
    except Exception as e:
        print(f"\n[ERRO] Falha ao carregar os pesos do arquivo: {e}")
        return None


def main():
    """Função principal que gerencia o fluxo do programa."""
    best_nn_weights = None

    while True:
        print("\n" + "=" * 54)
        print("   T2 – Aprendizagem por Reforço: RN + AG + Minimax")
        print("=" * 54)

        print("\n1. Jogar contra o Minimax (Você = X)")
        print("2. Treinar a Rede Neural (AG vs. Minimax)")
        print("3. Jogar contra a Rede Neural Treinada (Você = O)")
        print("4. Sair")

        choice = input("\nInforme a opção desejada (1-4): ").strip()

        if choice == "1":
            play_user_vs_minimax()
        elif choice == "2":
            try:
                pop_size = int(
                    input(f"Tamanho da população (padrão: {DEFAULT_POP_SIZE}): ")
                    or DEFAULT_POP_SIZE
                )
                gens_medium = int(
                    input(
                        f"Nº de gerações no modo Médio (padrão: {DEFAULT_GENS_MEDIUM}): "
                    )
                    or DEFAULT_GENS_MEDIUM
                )
                gens_hard = int(
                    input(
                        f"Nº de gerações no modo Difícil (padrão: {DEFAULT_GENS_HARD}): "
                    )
                    or DEFAULT_GENS_HARD
                )
                slow_mode = (
                    input(
                        "Executar em modo lento (visualização)? (s/n) [padrão: n]: "
                    ).lower()
                    == "s"
                )

                best_nn_weights = train_nn_vs_minimax(
                    pop_size, gens_medium, gens_hard, slow_mode
                )

                save_weights_to_file(best_nn_weights)

            except ValueError:
                print("\n[ERRO] Entrada inválida. Usando valores padrão.")
                best_nn_weights = train_nn_vs_minimax(
                    DEFAULT_POP_SIZE, DEFAULT_GENS_MEDIUM, DEFAULT_GENS_HARD, False
                )
                save_weights_to_file(best_nn_weights)

        elif choice == "3":
            weights_to_use = load_weights_from_file()

            if weights_to_use is None:
                print(f"[INFO] Nenhum arquivo de pesos ('{WEIGHTS_FILE}') encontrado.")
                if best_nn_weights:
                    print("[INFO] Usando os pesos da sessão de treinamento atual.")
                    weights_to_use = best_nn_weights

            play_user_vs_trained_nn(weights_to_use)

        elif choice == "4":
            print("Saindo do programa.")
            break
        else:
            print("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main()
