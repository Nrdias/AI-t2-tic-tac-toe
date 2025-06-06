#!/usr/bin/env python3
"""
Persistent Weight Visualizer for Genetic Algorithm Neural Networks

This version saves the population state and allows continuing evolution
from where you left off.
"""

import numpy as np
from genetic import *
import pickle
import os

class PersistentWeightVisualizer:
    def __init__(self):
        self.pop = None
        self.current_generation = 0
        self.save_file = 'evolution_state.pkl'
        
    def save_state(self):
        """Save current population and generation to file."""
        if self.pop is not None:
            state = {
                'population': self.pop.copy(),
                'generation': self.current_generation
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
            print(f"📂 Loaded evolution state (Generation {self.current_generation})")
            return True
        except FileNotFoundError:
            print("ℹ️  No saved state found. Starting fresh.")
            return False
    
    def extract_weights(self, individual_weights):
        """Extract and reshape weights from flat vector into network structure."""
        assert len(individual_weights) == WEIGHTS_COLS
        
        w1_end = 81
        b1_end = w1_end + 9
        w2_end = b1_end + 81
        
        w1 = individual_weights[:w1_end].reshape(9, 9)
        b1 = individual_weights[w1_end:b1_end]
        w2 = individual_weights[b1_end:w2_end].reshape(9, 9) 
        b2 = individual_weights[w2_end:]
        
        return w1, b1, w2, b2
    
    def print_matrix(self, matrix, name, precision=3):
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
    
    def print_weight_summary(self, w1, b1, w2, b2):
        """Print summary statistics for weights."""
        all_weights = np.concatenate([w1.flatten(), b1, w2.flatten(), b2])
        
        print(f"\n📊 WEIGHT STATISTICS:")
        print(f"   Total parameters: {len(all_weights)}")
        print(f"   Weight range: [{all_weights.min():.3f}, {all_weights.max():.3f}]")
        print(f"   Mean: {all_weights.mean():.3f}")
        print(f"   Std deviation: {all_weights.std():.3f}")
        print(f"   Zeros: {np.sum(np.abs(all_weights) < 0.001)}")
    
    def visualize_individual(self, individual_idx, individual, show_weights=True, precision=3):
        """Visualize a single individual's neural network."""
        weights = individual[:WEIGHTS_COLS]
        fitness = individual[-1]
        
        w1, b1, w2, b2 = self.extract_weights(weights)
        
        print("=" * 80)
        print(f"🧠 NEURAL NETWORK #{individual_idx + 1}")
        print("=" * 80)
        print(f"Fitness: {fitness:.4f}")
        print(f"Network Architecture: 9 inputs → 9 hidden (tanh) → 9 outputs (linear)")
        
        if show_weights:
            # Layer 1: Input to Hidden
            print(f"\n🔗 LAYER 1: Input → Hidden")
            self.print_matrix(w1, "Weight Matrix W1 (9×9)", precision)
            self.print_matrix(b1, "Bias Vector B1 (9)", precision)
            
            # Layer 2: Hidden to Output  
            print(f"\n🔗 LAYER 2: Hidden → Output")
            self.print_matrix(w2, "Weight Matrix W2 (9×9)", precision)
            self.print_matrix(b2, "Bias Vector B2 (9)", precision)
            
            # Statistics
            self.print_weight_summary(w1, b1, w2, b2)
        
        print("=" * 80)
    
    def show_current_population(self, num_individuals=5, precision=2):
        """Show current population weights."""
        if self.pop is None:
            print("❌ No population loaded. Start evolution first.")
            return
        
        # Sort by fitness (best first)
        sorted_indices = np.argsort(self.pop[:, -1])
        
        print("🧬 CURRENT POPULATION WEIGHTS")
        print("=" * 80)
        print(f"Generation: {self.current_generation}")
        print(f"Population size: {len(self.pop)}")
        print(f"Showing top {num_individuals} individuals")
        print("=" * 80)
        
        for i in range(min(num_individuals, len(self.pop))):
            idx = sorted_indices[i]
            individual = self.pop[idx]
            self.visualize_individual(i, individual, show_weights=True, precision=precision)
            
            if i < num_individuals - 1:
                input("\nPress Enter to see next individual...")
    
    def continue_evolution(self, num_generations=5):
        """Continue evolution from current state."""
        if self.pop is None:
            print("🆕 No existing population. Creating new one...")
            self.pop = random_population()
            evaluate_fitness(self.pop)
            self.current_generation = 0
        
        print(f"🚀 Continuing evolution from generation {self.current_generation}")
        print(f"Running {num_generations} more generations...")
        
        for i in range(num_generations):
            # Sort population by fitness
            self.pop = self.pop[np.argsort(self.pop[:, -1])]
            best_fitness = self.pop[0, -1]
            
            self.current_generation += 1
            print(f"Generation {self.current_generation}: Best fitness = {best_fitness:.4f}")
            
            # Check if target reached
            if best_fitness <= TARGET_FITNESS:
                print(f"🎯 Target fitness {TARGET_FITNESS} reached!")
                break
            
            # Create next generation
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
        
        # Auto-save after evolution
        self.save_state()
        
        return self.pop
    
    def reset_evolution(self):
        """Reset evolution state and start fresh."""
        if os.path.exists(self.save_file):
            os.remove(self.save_file)
        self.pop = None
        self.current_generation = 0
        print("🔄 Evolution state reset. Starting fresh next time.")

def main():
    """Main function with persistent evolution options."""
    print("🧠 Persistent Neural Network Weight Visualizer")
    print("=" * 60)
    print("1. Show current population weights")
    print("2. Continue evolution (add more generations)")
    print("3. Start new evolution (reset)")
    print("4. Load saved weights and analyze")
    print("5. Save current best individual")
    print("6. Exit")
    
    visualizer = PersistentWeightVisualizer()
    
    # Try to load existing state on startup
    visualizer.load_state()
    
    while True:
        try:
            if visualizer.pop is not None:
                best_fitness = np.min(visualizer.pop[:, -1])
                print(f"\n📊 Current state: Generation {visualizer.current_generation}, Best fitness: {best_fitness:.4f}")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                # Show current population
                if visualizer.pop is None:
                    print("❌ No population available. Start evolution first (option 2 or 3).")
                else:
                    num_show = int(input("How many individuals to show? (default 3): ") or "3")
                    precision = int(input("Weight precision (decimal places, default 2): ") or "2")
                    visualizer.show_current_population(num_show, precision)
                    
            elif choice == '2':
                # Continue evolution
                num_gens = int(input("How many generations to add? (default 5): ") or "5")
                visualizer.continue_evolution(num_gens)
                
            elif choice == '3':
                # Reset and start new
                confirm = input("Reset current evolution? (y/n): ")
                if confirm.lower().startswith('y'):
                    visualizer.reset_evolution()
                    num_gens = int(input("How many generations to run? (default 10): ") or "10")
                    visualizer.continue_evolution(num_gens)
                    
            elif choice == '4':
                # Load saved weights
                try:
                    weights = np.load('best_weights.npy')
                    print(f"Loaded weights with shape: {weights.shape}")
                    
                    individual = np.zeros(GENE_COLS)
                    individual[:WEIGHTS_COLS] = weights
                    individual[-1] = 0.0
                    
                    visualizer.visualize_individual(0, individual, show_weights=True, precision=3)
                    
                except FileNotFoundError:
                    print("No saved weights found.")
                    
            elif choice == '5':
                # Save best individual
                if visualizer.pop is None:
                    print("❌ No population available.")
                else:
                    sorted_pop = visualizer.pop[np.argsort(visualizer.pop[:, -1])]
                    best_weights = sorted_pop[0, :-1]  # Exclude fitness
                    np.save('persistent_best_weights.npy', best_weights)
                    print(f"💾 Best weights saved to 'persistent_best_weights.npy'")
                    print(f"   Fitness: {sorted_pop[0, -1]:.4f}")
                    
            elif choice == '6':
                # Save state before exit
                if visualizer.pop is not None:
                    visualizer.save_state()
                print("👋 Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            if visualizer.pop is not None:
                visualizer.save_state()
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 