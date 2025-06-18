import numpy as np
from src.game_environment.tictactoe import TicTacToe
from src.neural_network.mlp import MLP
from src.minimax.minimax_player import MinimaxPlayer
from src.config import (
    PLAYER_X, PLAYER_O, EMPTY_CELL, INPUT_NEURONS, HIDDEN_LAYER_1_NEURONS,
    OUTPUT_NEURONS
)

class ConsoleInterface:
    """
    Interface de usuário baseada em console para o Jogo da Velha.
    """
    def __init__(self, trained_nn_weights=None):
        self.trained_nn_weights = trained_nn_weights
        self.nn_player = None
        if self.trained_nn_weights is not None:
            self.nn_player = MLP(INPUT_NEURONS, HIDDEN_LAYER_1_NEURONS, OUTPUT_NEURONS)
            self.nn_player.set_all_weights(self.trained_nn_weights)
            
            # A função make_move_decision é injetada para que a MLP funcione como um "player"
            self.nn_player.make_move_decision = self._get_nn_move_decision_for_game

        self.minimax_player_hard = MinimaxPlayer(PLAYER_O, PLAYER_X, difficulty_mode="hard")
        self.game_instance = None
        self.current_game_mode = None

    def _get_nn_move_decision_for_game(self, board_state):
        """
        Função auxiliar para a rede neural decidir a jogada, garantindo que seja válida.
        Esta é a função que é chamada quando a NN é o player no objeto Game.
        """
        outputs = self.nn_player.forward_propagation(board_state)
        sorted_indices = np.argsort(outputs)[::-1] # Do maior para o menor (preferência)

        for idx in sorted_indices:
            r, c = divmod(idx, 3)
            if board_state[idx] == EMPTY_CELL: # Verifica se a célula está vazia ANTES de retornar
                return r, c
        return -1, -1 # Se nenhuma jogada válida for encontrada (muito raro em tabuleiro não cheio)

    def display_board(self, board_state):
        """
        Exibe o tabuleiro no console.
        """
        symbol_map = {PLAYER_X: 'X', PLAYER_O: 'O', EMPTY_CELL: '-'}
        print("\n--- TABULEIRO ---")
        for i in range(3):
            row_symbols = [symbol_map[board_state[i*3 + j]] for j in range(3)]
            print(" ".join(row_symbols))
        print("-----------------")

    def get_user_move(self):
        """
        Solicita e retorna a jogada do usuário.
        """
        while True:
            try:
                print("Sua vez. Digite a linha e coluna (0-2), ex: '0 0' para canto superior esquerdo:")
                user_input = input(">> ").strip()
                row, col = map(int, user_input.split())
                if 0 <= row <= 2 and 0 <= col <= 2:
                    return row, col
                else:
                    print("Entrada inválida. Linha e coluna devem ser entre 0 e 2.")
            except ValueError:
                print("Entrada inválida. Por favor, digite dois números separados por espaço.")

    def show_game_result(self, winner):
        """
        Exibe o resultado final do jogo.
        """
        print("\n--- FIM DE JOGO ---")
        if winner == PLAYER_X:
            print("O jogador X venceu!")
        elif winner == PLAYER_O:
            print("O jogador O venceu!")
        else:
            print("Empate!")
        print("-------------------\n")

    def play_game(self, mode):
        """
        Gerencia o fluxo de um jogo no console.
        """
        self.current_game_mode = mode
        self.game_instance = None

        if mode == "user_vs_minimax":
            print("\nIniciando: Você (X) vs Minimax (O - Difícil)")
            self.game_instance = TicTacToe(player1="Human", player2=self.minimax_player_hard) # Usuário é X
        elif mode == "user_vs_nn":
            if self.nn_player is None:
                print("Erro: Rede neural não treinada disponível para este modo.")
                return
            print("\nIniciando: Rede Neural (X) vs Você (O)")
            self.game_instance = TicTacToe(player1=self.nn_player, player2="Human") # Rede Neural é X
        else:
            print("Modo de jogo inválido.")
            return

        self.game_instance.reset_game()
        game_over = False

        while not game_over:
            self.display_board(self.game_instance.board.get_state())
            current_board_state = self.game_instance.board.get_state()

            current_player_symbol = self.game_instance.current_player
            made_move_success = False

            # Lógica para PLAYER_X
            if current_player_symbol == PLAYER_X:
                if (self.current_game_mode == "user_vs_minimax" and self.game_instance.player1 == "Human") or \
                   (self.current_game_mode == "user_vs_nn" and self.game_instance.player1 != "Human"): # Rede Neural é Player 1
                    # Vez do jogador humano ou da Rede Neural (PLAYER_X)
                    if self.game_instance.player1 == "Human":
                        row, col = self.get_user_move()
                        made_move_success = self.game_instance.board.make_move(row, col, current_player_symbol)
                        if not made_move_success:
                            print("Jogada inválida! Tente novamente.")
                            continue # Pede outra jogada ao usuário
                    else: # É a Rede Neural
                        print("Vez da Rede Neural (X)...")
                        # Chama make_move_decision da rede, que já está configurado para _get_nn_move_decision_for_game
                        row, col = self.game_instance.player1.make_move_decision(current_board_state)
                        made_move_success = self.game_instance.board.make_move(row, col, current_player_symbol)
                        if not made_move_success:
                            print("A Rede Neural tentou uma jogada inválida ou não encontrou uma válida! Fim de jogo.")
                            self.game_instance.game_over = True # Encerra o jogo por erro da IA
                            break

            # Lógica para PLAYER_O
            elif current_player_symbol == PLAYER_O:
                if (self.current_game_mode == "user_vs_minimax" and self.game_instance.player2 != "Human") or \
                   (self.current_game_mode == "user_vs_nn" and self.game_instance.player2 == "Human"): # Usuário é Player 2
                    # Vez do Minimax ou do jogador humano (PLAYER_O)
                    if self.game_instance.player2 == "Human":
                        row, col = self.get_user_move()
                        made_move_success = self.game_instance.board.make_move(row, col, current_player_symbol)
                        if not made_move_success:
                            print("Jogada inválida! Tente novamente.")
                            continue # Pede outra jogada ao usuário
                    else: # É o Minimax
                        print("Vez do Minimax (O)...")
                        row, col = self.game_instance.player2.make_move_decision(current_board_state)
                        made_move_success = self.game_instance.board.make_move(row, col, current_player_symbol)
                        if not made_move_success:
                            print("O Minimax tentou uma jogada inválida! Fim de jogo.")
                            self.game_instance.game_over = True # Encerra o jogo por erro do Minimax
                            break
            
            if made_move_success:
                self.game_instance._check_game_status()
                winner, game_over = self.game_instance.get_game_result()
                if not game_over:
                    self.game_instance._switch_player()
            else:
                # Se não conseguiu fazer a jogada (ex: player humano digita inválido e loop continua)
                pass # A mensagem de erro já foi impressa ou o loop será reiniciado

        self.display_board(self.game_instance.board.get_state())
        self.show_game_result(self.game_instance.winner)


    def calculate_and_display_accuracy(self, num_test_games=100):
        """
        Calcula a acurácia da rede treinada jogando contra o Minimax no console.
        """
        if not self.nn_player:
            print("\nRede neural não disponível para teste de acurácia.")
            return

        print(f"\n--- TESTANDO ACURÁCIA DA REDE NEURAL (X) vs MINIMAX (O - Difícil) - {num_test_games} JOGOS ---")

        wins = 0
        losses = 0
        draws = 0
        invalid_moves = 0

        test_minimax_opponent = MinimaxPlayer(PLAYER_O, PLAYER_X, difficulty_mode="hard")

        # Para o teste de acurácia, a Rede Neural sempre será o Player X
        game_for_accuracy_test = TicTacToe(player1=self.nn_player, player2=test_minimax_opponent)

        for i in range(num_test_games):
            game_for_accuracy_test.reset_game()

            while not game_for_accuracy_test.game_over:
                # O play_turn na classe Game é responsável por alternar os jogadores
                # e chamar o make_move_decision apropriado (NN ou Minimax)
                is_valid_move = game_for_accuracy_test.play_turn()
                
                # Se o Player X (Rede Neural) fez uma jogada inválida, penaliza e encerra o jogo
                if not is_valid_move and game_for_accuracy_test.current_player == PLAYER_X:
                    invalid_moves += 1
                    game_for_accuracy_test.game_over = True
                    break

            winner, _ = game_for_accuracy_test.get_game_result()
            if winner == PLAYER_X:
                wins += 1
            elif winner == PLAYER_O:
                losses += 1
            elif winner == EMPTY_CELL:
                draws += 1
            
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{num_test_games} jogos testados.")

        total_games_considered = wins + losses + draws
        if total_games_considered > 0:
            accuracy = (wins / total_games_considered) * 100
        else:
            accuracy = 0.0

        print(f"\n--- RESULTADOS DO TESTE DE ACURÁCIA ---")
        print(f"Total de jogos válidos: {total_games_considered}")
        print(f"Vitórias da Rede Neural: {wins}")
        print(f"Derrotas da Rede Neural: {losses}")
        print(f"Empates: {draws}")
        if invalid_moves > 0:
            print(f"Jogadas inválidas pela Rede Neural (jogos encerrados): {invalid_moves}")
        print(f"Acurácia da Rede Neural (Vitórias sobre jogos válidos): {accuracy:.2f}%")
        print("-------------------------------------------\n")