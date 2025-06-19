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
    "in_progress": 5,
    "prioritary": 1,
    "offensive": 2,
    "defensive": 3,
    "draw": 20,
    "defeat": -5,
    "victory": 10,
}

# Arquivo com melhores pesos da rede neural
WEIGHTS_FILE = "best_nn_weights.csv"

# Parâmetros do algoritmo genético
DEFAULT_POP_SIZE = 20
DEFAULT_GENS_MEDIUM = 10
DEFAULT_GENS_HARD = 20
