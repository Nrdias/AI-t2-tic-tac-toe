# Genetic Algorithm to evolve a simple NN that plays Tic‑Tac‑Toe
# Author: (adapted for Natan Dias, 2025‑06‑04)

"""
Overview
========
Each individual is a vector with **181 columns**:
    0 … 179 ‑‑ floating‑point weights in the interval [‑1, 1]
          (they parameterise a 2‑layer feed‑forward network)
    180    ‑‑ fitness (the lower, the better)

Network architecture
--------------------
* 9 input cells (board squares, ‑1=O, 0=empty, 1=X)
* 9 hidden neurons (tanh)
* 9 output neurons (linear) → argmax is the selected move

The 180 weights are laid out like this::
    w1  (9×9)  =  81
    b1  (9)    =   9   (running total  90)
    w2  (9×9)  =  81   (running total 171)
    b2  (9)    =   9   (running total 180)

Fitness
-------
Each individual plays *GAMES_PER_EVAL* matches as **player X** against
an ε‑random opponent (90 % best‑move, 10 % random).  Score per match::
    win  →  0   (ideal)
    draw →  1
    loss →  2
The fitness stored in col‑180 is the mean score across the matches, so
**lower is better**.

GA hyper‑parameters (tuned for a quick demo; adjust as desired):
---------------------------------------------------------------
POP_SIZE          = 120
ELITE_SIZE        =   4   # how many individuals survive untouched
MAX_GENERATIONS   = 600
MUT_RATE          = 0.08  # per‑gene probability
MUT_SIGMA         = 0.15  # std‑dev of Gaussian noise

Crossover: *whole‑arithmetic (blend)*::
    c1 = α·p1 + (1‑α)·p2
    c2 = (1‑α)·p1 + α·p2
with α ~ U(0, 1) drawn anew for every pair.

Requires *numpy* only.  Run the file to watch the evolution.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# ---------- GA Hyper‑parameters ----------
POP_SIZE        = 120
ELITE_SIZE      = 4
MAX_GENERATIONS = 600
MUT_RATE        = 0.08     # probability of mutating each weight
MUT_SIGMA       = 0.15     # std‑dev for Gaussian perturbation
GAMES_PER_EVAL  = 30
TARGET_FITNESS  = 0.40      # stop if reached (≈ 80 % wins)

WEIGHTS_COLS    = 180
GENE_COLS       = WEIGHTS_COLS + 1  # +1 for fitness

# ---------- Tic‑Tac‑Toe Environment ----------

LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),        # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),        # cols
    (0, 4, 8), (2, 4, 6)                    # diags
]

@dataclass
class Board:
    cells: np.ndarray  # shape (9,) values in {‑1, 0, 1}

    @classmethod
    def new(cls):
        return cls(np.zeros(9, dtype=float))

    def legal_moves(self) -> List[int]:
        return [i for i, v in enumerate(self.cells) if v == 0]

    def play(self, idx: int, mark: float):
        self.cells[idx] = mark

    def winner(self) -> float | None:
        for a, b, c in LINES:
            s = self.cells[[a, b, c]].sum()
            if s == 3:
                return 1  # X wins
            if s == ‑3:
                return ‑1 # O wins
        if not self.legal_moves():
            return 0  # draw
        return None  # game continues

# ---------- Simple ε‑greedy opponent ----------

class HeuristicOpponent:
    """Plays winning moves, blocks immediate threats, otherwise random."""

    def __init__(self, eps: float = 0.1):
        self.eps = eps

    def choose(self, board: Board, mark: float) -> int:
        moves = board.legal_moves()
        if random.random() < self.eps:
            return random.choice(moves)
        # 1) winning move
        for m in moves:
            b = board.cells.copy(); b[m] = mark
            if Board(b).winner() == mark:
                return m
        # 2) block opponent
        opp = ‑mark
        for m in moves:
            b = board.cells.copy(); b[m] = opp
            if Board(b).winner() == opp:
                return m
        # 3) otherwise centre, corners, sides preference
        for p in [4, 0, 2, 6, 8, 1, 3, 5, 7]:
            if p in moves:
                return p
        return random.choice(moves)

# ---------- Neural Network player ----------

class Net:
    """2‑layer fully‑connected NN with tanh hidden, linear output."""

    def __init__(self, weights: np.ndarray):
        assert len(weights) == WEIGHTS_COLS
        # split flat vector into matrices / biases according to layout
        w1_end = 81
        b1_end = w1_end + 9
        w2_end = b1_end + 81

        w1 = weights[:w1_end].reshape(9, 9)
        b1 = weights[w1_end:b1_end]
        w2 = weights[b1_end:w2_end].reshape(9, 9)
        b2 = weights[w2_end:]
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

    def policy(self, board: Board) -> int:
        x = board.cells.copy()
        h = np.tanh(x @ self.w1 + self.b1)
        y = h @ self.w2 + self.b2
        # mask illegal moves to ‑inf
        y_masked = np.where(board.cells == 0, y, ‑np.inf)
        if np.all(np.isneginf(y_masked)):
            return random.choice(board.legal_moves())
        return int(np.argmax(y_masked))

# ---------- GA Operators ----------

def random_population() -> np.ndarray:
    pop = np.empty((POP_SIZE, GENE_COLS), dtype=float)
    pop[:, :WEIGHTS_COLS] = np.random.uniform(‑1, 1, size=(POP_SIZE, WEIGHTS_COLS))
    pop[:, ‑1] = np.inf  # fitness initially unknown
    return pop


def evaluate_fitness(pop: np.ndarray):
    opp = HeuristicOpponent(eps=0.1)
    for idx in range(POP_SIZE):
        w = pop[idx, :WEIGHTS_COLS]
        net = Net(w)
        score = 0.0
        for _ in range(GAMES_PER_EVAL):
            score += play_single_game(net, opp)
        pop[idx, ‑1] = score / GAMES_PER_EVAL


def play_single_game(net: Net, opponent: HeuristicOpponent) -> float:
    board = Board.new()
    turn = 0
    while True:
        if turn % 2 == 0:  # Net plays X (1)
            move = net.policy(board)
            board.play(move, 1)
        else:
            move = opponent.choose(board, ‑1)
            board.play(move, ‑1)
        w = board.winner()
        if w is not None:
            # convert result to fitness increment (lower is better)
            return {1: 0.0, 0: 1.0, ‑1: 2.0}[w]
        turn += 1


def tournament(pop: np.ndarray) -> int:
    i, j = random.sample(range(POP_SIZE), 2)
    return i if pop[i, ‑1] < pop[j, ‑1] else j


def crossover(parents: np.ndarray, offspring: np.ndarray):
    assert offspring.shape == parents.shape
    # Elites already copied outside; start filling from ELITE_SIZE
    for k in range(ELITE_SIZE, POP_SIZE, 2):
        p1 = parents[tournament(parents)]
        p2 = parents[tournament(parents)]
        alpha = random.random()
        child1 = alpha * p1[:WEIGHTS_COLS] + (1‑alpha) * p2[:WEIGHTS_COLS]
        child2 = (1‑alpha) * p1[:WEIGHTS_COLS] + alpha * p2[:WEIGHTS_COLS]
        offspring[k, :WEIGHTS_COLS]   = child1
        offspring[k+1, :WEIGHTS_COLS] = child2


def mutate(pop: np.ndarray):
    mask = np.random.rand(POP_SIZE, WEIGHTS_COLS) < MUT_RATE
    noise = np.random.normal(0, MUT_SIGMA, size=(POP_SIZE, WEIGHTS_COLS))
    pop[:, :WEIGHTS_COLS] += mask * noise
    np.clip(pop[:, :WEIGHTS_COLS], ‑1, 1, out=pop[:, :WEIGHTS_COLS])


# ---------- Main loop ----------

def ga_run():
    pop = random_population()
    evaluate_fitness(pop)

    for gen in range(1, MAX_GENERATIONS + 1):
        pop = pop[np.argsort(pop[:, ‑1])]  # sort ascending (best first)
        best = pop[0, ‑1]
        print(f"Gen {gen:>3}  best fitness = {best:.3f}")
        if best <= TARGET_FITNESS:
            break
        # Elitism: copy top‑ELITE_SIZE to next generation
        new_pop = np.empty_like(pop)
        new_pop[:ELITE_SIZE] = pop[:ELITE_SIZE].copy()
        # Crossover fills the rest (overwrites fitness cols)
        crossover(pop, new_pop)
        # Mutation (does not touch fitness col)
        mutate(new_pop)
        # Mark fitness unknown for non‑elites
        new_pop[ELITE_SIZE:, ‑1] = np.inf
        pop = new_pop
        evaluate_fitness(pop)

    print("--- Finished ---")
    pop = pop[np.argsort(pop[:, ‑1])]
    print(f"Best individual fitness = {pop[0, ‑1]:.3f}")
    return pop[0]


if __name__ == "__main__":
    ga_run()
