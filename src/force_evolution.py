#!/usr/bin/env python3
"""
Forced Evolution Visualizer - Guarantees multiple generations of evolution
by removing early stopping and making the task more challenging.
"""

import numpy as np
from genetic import *
import pickle
import os

class ForcedEvolutionVisualizer:
    def __init__(self):
        self.pop = None
        self.current_generation = 0
        self.save_file = 'forced_evolution_state.pkl'
        self.generation_history = []  # Track fitness over generations
        
    def save_state(self):
        """Save current population and generation to file."""
        if self.pop is not None:
            state = {
                'population': self.pop.copy(),
                'generation': self.current_generation,
                'history': self.generation_history
            }
            with open(self.save_file, 'wb') as f:
                pickle.dump(state, f)
            print(f"💾 Saved evolution state (Generation {self.current_generation})")
    
    def load_state(self):
        """Load population and generation from file."""
        try:
            with open(self.save_file, 'rb') as f:
                state = pickle.load(f)
            self.pop = state['population']
            self.current_generation = state['generation']
            self.generation_history = state.get('history', [])
            print(f"📂 Loaded evolution state (Generation {self.current_generation})")
            return True
        except FileNotFoundError:
            print("ℹ️  No saved state found. Starting fresh.")
            return False
    
    def extract_weights(self, individual_weights):
        """Extract and reshape weights from flat vector into network structure."""
        w1_end = 81
        b1_end = w1_end + 9
        w2_end = b1_end + 81
        
        w1 = individual_weights[:w1_end].reshape(9, 9)
        b1 = individual_weights[w1_end:b1_end]
        w2 = individual_weights[b1_end:w2_end].reshape(9, 9) 
        b2 = individual_weights[w2_end:]
        
        return w1, b1, w2, b2
    
    def print_matrix(self, matrix, name, precision=2):
        """Print a matrix with nice formatting."""
        print(f"\n{name}:")
        print("-" * (len(name) + 1))
        
        if matrix.ndim == 1:  # Vector (bias)
            print("  [", end="")
            for i, val in enumerate(matrix):
                if i > 0:
                    print(", ", end="")
                print(f"{val:+.{precision}f}", end="")
            print("]")
        else:  # Matrix
            rows, cols = matrix.shape
            for i in range(rows):
                print("  [", end="")
                for j in range(cols):
                    if j > 0:
                        print(", ", end="")
                    print(f"{matrix[i,j]:+.{precision}f}", end="")
                print("]" if i == rows-1 else "],")
    
    def show_individual_weights(self, individual_idx, individual, precision=2):
        """Show weights for a single individual."""
        weights = individual[:WEIGHTS_COLS]
        fitness = individual[-1]
        
        w1, b1, w2, b2 = self.extract_weights(weights)
        
        print(f"\n🧠 INDIVIDUAL #{individual_idx + 1} (Fitness: {fitness:.4f})")
        print("=" * 60)
        
        # Layer 1: Input to Hidden
        self.print_matrix(w1, "W1 (Input→Hidden)", precision)
        self.print_matrix(b1, "B1 (Hidden Bias)", precision)
        
        # Layer 2: Hidden to Output  
        self.print_matrix(w2, "W2 (Hidden→Output)", precision)
        self.print_matrix(b2, "B2 (Output Bias)", precision)
        
        # Statistics
        all_weights = np.concatenate([w1.flatten(), b1, w2.flatten(), b2])
        print(f"\n📊 WEIGHT STATS: Range [{all_weights.min():.3f}, {all_weights.max():.3f}], Mean: {all_weights.mean():.3f}")
    
    def forced_evolution(self, num_generations=20, games_per_eval=50):
        """Run evolution for a fixed number of generations (no early stopping)."""
        
        # Make it more challenging
        global GAMES_PER_EVAL
        original_games = GAMES_PER_EVAL
        GAMES_PER_EVAL = games_per_eval
        
        if self.pop is None:
            print("🆕 Creating new population...")
            self.pop = random_population()
            evaluate_fitness(self.pop)
            self.current_generation = 0
            self.generation_history = []
        
        print(f"🚀 FORCED EVOLUTION - {num_generations} generations guaranteed!")
        print(f"📊 Using {GAMES_PER_EVAL} games per evaluation for accuracy")
        print("🚫 Early stopping DISABLED - will run all generations")
        print("-" * 70)
        
        try:
            for i in range(num_generations):
                # Sort population by fitness
                self.pop = self.pop[np.argsort(self.pop[:, -1])]
                best_fitness = self.pop[0, -1]
                avg_fitness = np.mean(self.pop[:, -1])
                worst_fitness = self.pop[-1, -1]
                
                self.current_generation += 1
                
                # Record generation stats
                gen_stats = {
                    'generation': self.current_generation,
                    'best': best_fitness,
                    'avg': avg_fitness,
                    'worst': worst_fitness
                }
                self.generation_history.append(gen_stats)
                
                print(f"Gen {self.current_generation:2d}: Best={best_fitness:.4f} Avg={avg_fitness:.4f} Worst={worst_fitness:.4f}")
                
                # Show top 3 individuals every 5 generations
                if self.current_generation % 5 == 0 or i == 0:
                    print(f"\n📋 TOP 3 INDIVIDUALS (Generation {self.current_generation}):")
                    for j in range(3):
                        individual = self.pop[j]
                        weights = individual[:WEIGHTS_COLS]
                        fitness = individual[-1]
                        w1, b1, w2, b2 = self.extract_weights(weights)
                        all_weights = np.concatenate([w1.flatten(), b1, w2.flatten(), b2])
                        
                        print(f"  #{j+1}: Fitness={fitness:.4f}, Weight range=[{all_weights.min():.2f},{all_weights.max():.2f}], Mean={all_weights.mean():.3f}")
                    print()
                
                # Create next generation (NO EARLY STOPPING)
                new_pop = np.empty_like(self.pop)
                new_pop[:ELITE_SIZE] = self.pop[:ELITE_SIZE].copy()  # Elitism
                
                # Crossover and mutation
                crossover(self.pop, new_pop)
                mutate(new_pop)
                
                # Mark fitness unknown for non-elites
                new_pop[ELITE_SIZE:, -1] = np.inf
                self.pop = new_pop
                
                # Evaluate fitness
                evaluate_fitness(self.pop)
                
        except KeyboardInterrupt:
            print("\n⏹️ Evolution stopped by user")
        
        finally:
            # Restore original settings
            GAMES_PER_EVAL = original_games
        
        # Auto-save after evolution
        self.save_state()
        print(f"\n🏁 Completed {self.current_generation} generations of forced evolution!")
        
        return self.pop
    
    def show_evolution_history(self):
        """Show how fitness evolved over generations."""
        if not self.generation_history:
            print("❌ No evolution history available.")
            return
        
        print("\n📈 EVOLUTION HISTORY")
        print("=" * 60)
        print("Gen |   Best   |   Avg    |  Worst   | Improvement")
        print("-" * 60)
        
        for i, stats in enumerate(self.generation_history):
            gen = stats['generation']
            best = stats['best']
            avg = stats['avg']
            worst = stats['worst']
            
            improvement = ""
            if i > 0:
                prev_best = self.generation_history[i-1]['best']
                change = prev_best - best  # Lower is better
                if change > 0.001:
                    improvement = f"↑{change:.3f}"
                elif change < -0.001:
                    improvement = f"↓{abs(change):.3f}"
                else:
                    improvement = "→"
            
            print(f"{gen:3d} | {best:8.4f} | {avg:8.4f} | {worst:8.4f} | {improvement:>10}")
    
    def show_detailed_weights(self, num_individuals=3):
        """Show detailed weights for top individuals."""
        if self.pop is None:
            print("❌ No population available.")
            return
        
        sorted_pop = self.pop[np.argsort(self.pop[:, -1])]
        
        print(f"\n🔍 DETAILED WEIGHTS - TOP {num_individuals} INDIVIDUALS")
        print("=" * 80)
        
        for i in range(min(num_individuals, len(sorted_pop))):
            self.show_individual_weights(i, sorted_pop[i], precision=2)
            if i < num_individuals - 1:
                input("\nPress Enter to see next individual...")

def main():
    """Main function for forced evolution."""
    print("🔥 FORCED EVOLUTION VISUALIZER")
    print("=" * 50)
    print("This version GUARANTEES evolution over multiple generations!")
    print("1. Run forced evolution (20 generations)")
    print("2. Show current population weights")
    print("3. Show evolution history")
    print("4. Continue evolution (add more generations)")
    print("5. Reset and start new")
    print("6. Exit")
    
    visualizer = ForcedEvolutionVisualizer()
    visualizer.load_state()
    
    while True:
        try:
            if visualizer.pop is not None:
                best_fitness = np.min(visualizer.pop[:, -1])
                print(f"\n📊 Current: Gen {visualizer.current_generation}, Best fitness: {best_fitness:.4f}")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                num_gens = int(input("Number of generations? (default 20): ") or "20")
                games = int(input("Games per evaluation? (default 50): ") or "50")
                visualizer.forced_evolution(num_gens, games)
                
            elif choice == '2':
                num_show = int(input("How many individuals? (default 3): ") or "3")
                visualizer.show_detailed_weights(num_show)
                
            elif choice == '3':
                visualizer.show_evolution_history()
                
            elif choice == '4':
                if visualizer.pop is None:
                    print("❌ No population. Use option 1 first.")
                else:
                    num_gens = int(input("Additional generations? (default 10): ") or "10")
                    games = int(input("Games per evaluation? (default 50): ") or "50")
                    visualizer.forced_evolution(num_gens, games)
                    
            elif choice == '5':
                confirm = input("Reset evolution? (y/n): ")
                if confirm.lower().startswith('y'):
                    if os.path.exists(visualizer.save_file):
                        os.remove(visualizer.save_file)
                    visualizer.pop = None
                    visualizer.current_generation = 0
                    visualizer.generation_history = []
                    print("🔄 Reset complete!")
                    
            elif choice == '6':
                if visualizer.pop is not None:
                    visualizer.save_state()
                print("👋 Goodbye!")
                break
                
            else:
                print("Invalid choice.")
                
        except KeyboardInterrupt:
            if visualizer.pop is not None:
                visualizer.save_state()
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 