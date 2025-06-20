# Símbolos do tabuleiro (NN = NeuralNetwork)
NN_PLAYER_SYMBOL = "X"
OPPONENT_PLAYER_SYMBOL = "O"
EMPTY_CELL = " "

# Representação numérica para a rede neural
NN_PLAYER_VAL = 1
OPPONENT_PLAYER_VAL = -1
EMPTY_VAL = 0

# Pesos de fitness para as jogadas
FITNESS_WEIGHTS = {
    "invalid_other_pos": -30,
    "invalid_own_pos": -50,
    "in_progress": 3,
    "prioritary": 2,
    "offensive": 5,
    "defensive": 10,
    "draw": 25,
    "defeat": -15,
    "victory": 30,
}

# Arquivo com melhores pesos da rede neural
WEIGHTS_FILE = "best_nn_weights.csv"
WEIGHTS_FILE_MEDIUM = "best_nn_weights_medium.csv"
WEIGHTS_FILE_HARD = "best_nn_weights_hard.csv"

# Parâmetros do algoritmo genético
DEFAULT_POP_SIZE = 40
DEFAULT_GENS_MEDIUM = 15
DEFAULT_GENS_HARD = 40
