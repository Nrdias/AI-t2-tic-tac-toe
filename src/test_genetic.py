#!/usr/bin/env python3
"""
Test script for the Genetic Algorithm Tic-Tac-Toe player.

This script demonstrates several ways to test the genetic algorithm:
1. Run the evolution process
2. Test an evolved player against different opponents
3. Visualize game results
4. Manual testing interface
"""

import numpy as np
from genetic import *
import time

def print_board(board: Board):
    """Pretty print a tic-tac-toe board."""
    symbols = {-1: 'O', 0: '.', 1: 'X'}
    cells = [symbols[int(x)] for x in board.cells]
    print(f"\n {cells[0]} | {cells[1]} | {cells[2]} ")
    print("---|---|---")
    print(f" {cells[3]} | {cells[4]} | {cells[5]} ")
    print("---|---|---")  
    print(f" {cells[6]} | {cells[7]} | {cells[8]} \n")

def play_human_vs_ai(ai_weights: np.ndarray):
    """Let a human play against the evolved AI."""
    ai_net = Net(ai_weights)
    board = Board.new()
    
    print("You are O, AI is X. Enter moves as 0-8 (top-left to bottom-right)")
    print_board(board)
    
    while board.winner() is None:
        # Human turn (O = -1)
        try:
            human_move = int(input("Your move (0-8): "))
            if human_move not in board.legal_moves():
                print("Invalid move! Try again.")
                continue
            board.play(human_move, -1)
            print_board(board)
            
            if board.winner() is not None:
                break
                
            # AI turn (X = 1)
            ai_move = ai_net.policy(board)
            print(f"AI plays: {ai_move}")
            board.play(ai_move, 1)
            print_board(board)
            
        except (ValueError, KeyboardInterrupt):
            print("Game ended.")
            break
    
    winner = board.winner()
    if winner == 1:
        print("AI (X) wins!")
    elif winner == -1:
        print("You (O) win!")
    else:
        print("It's a draw!")

def test_ai_vs_random(ai_weights: np.ndarray, num_games: int = 100):
    """Test the AI against a random player."""
    ai_net = Net(ai_weights)
    
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(num_games):
        board = Board.new()
        turn = 0
        
        while board.winner() is None:
            if turn % 2 == 0:  # AI plays X
                move = ai_net.policy(board)
                board.play(move, 1)
            else:  # Random player plays O
                moves = board.legal_moves()
                move = np.random.choice(moves)
                board.play(move, -1)
            turn += 1
        
        result = board.winner()
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
    
    print(f"\nResults against random player ({num_games} games):")
    print(f"Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_games*100:.1f}%)")
    
    return wins, draws, losses

def test_ai_vs_heuristic(ai_weights: np.ndarray, num_games: int = 100):
    """Test the AI against the heuristic opponent."""
    ai_net = Net(ai_weights)
    opponent = HeuristicOpponent(eps=0.0)  # Perfect heuristic player
    
    wins = 0
    draws = 0 
    losses = 0
    
    for _ in range(num_games):
        board = Board.new()
        turn = 0
        
        while board.winner() is None:
            if turn % 2 == 0:  # AI plays X
                move = ai_net.policy(board)
                board.play(move, 1)
            else:  # Heuristic player plays O
                move = opponent.choose(board, -1)
                board.play(move, -1)
            turn += 1
        
        result = board.winner()
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
    
    print(f"\nResults against heuristic player ({num_games} games):")
    print(f"Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_games*100:.1f}%)")
    
    return wins, draws, losses

def run_evolution_test():
    """Run the genetic algorithm and test the best individual."""
    print("Starting genetic algorithm evolution...")
    print("This may take several minutes...")
    
    start_time = time.time()
    best_individual = ga_run()
    evolution_time = time.time() - start_time
    
    print(f"\nEvolution completed in {evolution_time:.1f} seconds")
    print(f"Best fitness: {best_individual[-1]:.3f}")
    
    # Extract weights (exclude fitness)
    best_weights = best_individual[:-1]
    
    # Test the evolved player
    print("\n" + "="*50)
    print("TESTING THE EVOLVED PLAYER")
    print("="*50)
    
    test_ai_vs_random(best_weights, 100)
    test_ai_vs_heuristic(best_weights, 100)
    
    return best_weights

def quick_test():
    """Quick test with fewer generations for debugging."""
    print("Running quick test (fewer generations)...")
    
    # Temporarily reduce parameters for quick testing
    global MAX_GENERATIONS, GAMES_PER_EVAL
    original_gens = MAX_GENERATIONS
    original_games = GAMES_PER_EVAL
    
    MAX_GENERATIONS = 50
    GAMES_PER_EVAL = 10
    
    try:
        best_weights = run_evolution_test()
        return best_weights
    finally:
        # Restore original values
        MAX_GENERATIONS = original_gens
        GAMES_PER_EVAL = original_games

def main():
    """Main testing interface."""
    print("Genetic Algorithm Tic-Tac-Toe Tester")
    print("="*40)
    print("1. Run full evolution test")
    print("2. Run quick test (fewer generations)")
    print("3. Load and test pre-evolved weights")
    print("4. Play against AI (requires evolved weights)")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                best_weights = run_evolution_test()
                # Optionally save the weights
                np.save('best_weights.npy', best_weights)
                print("Weights saved to 'best_weights.npy'")
                
            elif choice == '2':
                best_weights = quick_test()
                
            elif choice == '3':
                try:
                    weights = np.load('best_weights.npy')
                    print(f"Loaded weights with shape: {weights.shape}")
                    test_ai_vs_random(weights, 100)
                    test_ai_vs_heuristic(weights, 100)
                except FileNotFoundError:
                    print("No saved weights found. Run evolution first.")
                    
            elif choice == '4':
                try:
                    weights = np.load('best_weights.npy')
                    play_human_vs_ai(weights)
                except FileNotFoundError:
                    print("No saved weights found. Run evolution first.")
                    
            elif choice == '5':
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 