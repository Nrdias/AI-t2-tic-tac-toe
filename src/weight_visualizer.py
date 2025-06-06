#!/usr/bin/env python3
"""
Weight Visualizer for Genetic Algorithm Neural Networks

This script displays the weight matrices and biases for each individual
in the population, showing how the neural networks are structured.
"""

import numpy as np
from genetic import *
import sys

class WeightVisualizer:
    def __init__(self):
        self.pop = None
        
    def extract_weights(self, individual_weights):
        """Extract and reshape weights from flat vector into network structure."""
        assert len(individual_weights) == WEIGHTS_COLS
        
        # Extract weights according to the layout:
        # w1 (9×9) = 81 weights
        # b1 (9) = 9 biases  
        # w2 (9×9) = 81 weights
        # b2 (9) = 9 biases
        
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
    
    def visualize_population(self, pop, num_individuals=5, show_weights=True, precision=2):
        """Visualize multiple individuals from the population."""
        self.pop = pop
        
        # Sort by fitness (best first)
        sorted_indices = np.argsort(pop[:, -1])
        
        print("🧬 GENETIC ALGORITHM POPULATION WEIGHTS")
        print("=" * 80)
        print(f"Population size: {len(pop)}")
        print(f"Showing top {num_individuals} individuals")
        print(f"Weight precision: {precision} decimal places")
        print("=" * 80)
        
        for i in range(min(num_individuals, len(pop))):
            idx = sorted_indices[i]
            individual = pop[idx]
            self.visualize_individual(i, individual, show_weights, precision)
            
            if i < num_individuals - 1:
                input("\nPress Enter to see next individual...")
    
    def compare_individuals(self, pop, individual_indices, precision=3):
        """Compare weights between specific individuals."""
        print("⚖️  WEIGHT COMPARISON")
        print("=" * 80)
        
        individuals_data = []
        for i, idx in enumerate(individual_indices):
            individual = pop[idx]
            weights = individual[:WEIGHTS_COLS]
            fitness = individual[-1]
            w1, b1, w2, b2 = self.extract_weights(weights)
            
            individuals_data.append({
                'idx': idx,
                'fitness': fitness,
                'weights': weights,
                'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2
            })
            
            print(f"Individual #{idx}: fitness = {fitness:.4f}")
        
        print("\n🔍 WEIGHT DIFFERENCES:")
        
        if len(individuals_data) == 2:
            # Compare two individuals
            diff = individuals_data[0]['weights'] - individuals_data[1]['weights']
            print(f"\nWeight difference (Individual #{individual_indices[0]} - Individual #{individual_indices[1]}):")
            print(f"   Max difference: {np.abs(diff).max():.4f}")
            print(f"   Mean absolute difference: {np.abs(diff).mean():.4f}")
            print(f"   Correlation: {np.corrcoef(individuals_data[0]['weights'], individuals_data[1]['weights'])[0,1]:.4f}")
        
        # Show weight statistics for each
        for i, data in enumerate(individuals_data):
            print(f"\nIndividual #{data['idx']} statistics:")
            self.print_weight_summary(data['w1'], data['b1'], data['w2'], data['b2'])
    
    def visualize_weight_evolution(self, generations_data):
        """Visualize how weights change over generations."""
        print("📈 WEIGHT EVOLUTION OVER GENERATIONS")
        print("=" * 80)
        
        for gen_idx, (generation, pop) in enumerate(generations_data):
            best_individual = pop[np.argmin(pop[:, -1])]
            weights = best_individual[:WEIGHTS_COLS]
            fitness = best_individual[-1]
            
            print(f"\nGeneration {generation}: Best fitness = {fitness:.4f}")
            
            # Show weight statistics
            w1, b1, w2, b2 = self.extract_weights(weights)
            all_weights = np.concatenate([w1.flatten(), b1, w2.flatten(), b2])
            print(f"   Weight range: [{all_weights.min():.3f}, {all_weights.max():.3f}]")
            print(f"   Weight std: {all_weights.std():.3f}")

def main():
    """Main function with options for weight visualization."""
    print("🧠 Neural Network Weight Visualizer")
    print("=" * 50)
    print("1. Generate new population and show weights")
    print("2. Run evolution and show weight progression")
    print("3. Load saved weights and analyze")
    print("4. Compare individuals")
    print("5. Exit")
    
    visualizer = WeightVisualizer()
    
    while True:
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                # Generate new population
                print("Generating random population...")
                pop = random_population()
                evaluate_fitness(pop)
                
                num_show = int(input("How many individuals to show? (default 3): ") or "3")
                precision = int(input("Weight precision (decimal places, default 2): ") or "2")
                show_weights = input("Show full weight matrices? (y/n, default y): ").lower().startswith('y') if input else True
                
                visualizer.visualize_population(pop, num_show, show_weights, precision)
                
            elif choice == '2':
                # Run evolution and track weights
                print("Running evolution...")
                generations_data = []
                
                pop = random_population()
                evaluate_fitness(pop)
                generations_data.append((0, pop.copy()))
                
                for gen in range(1, 6):  # Just 5 generations for demo
                    pop = pop[np.argsort(pop[:, -1])]
                    best_fitness = pop[0, -1]
                    print(f"Generation {gen}: Best fitness = {best_fitness:.4f}")
                    
                    # Evolution step
                    new_pop = np.empty_like(pop)
                    new_pop[:ELITE_SIZE] = pop[:ELITE_SIZE].copy()
                    crossover(pop, new_pop)
                    mutate(new_pop)
                    new_pop[ELITE_SIZE:, -1] = np.inf
                    pop = new_pop
                    evaluate_fitness(pop)
                    
                    generations_data.append((gen, pop.copy()))
                
                visualizer.visualize_weight_evolution(generations_data)
                
            elif choice == '3':
                try:
                    weights = np.load('best_weights.npy')
                    print(f"Loaded weights with shape: {weights.shape}")
                    
                    # Create individual with loaded weights
                    individual = np.zeros(GENE_COLS)
                    individual[:WEIGHTS_COLS] = weights
                    individual[-1] = 0.0  # Unknown fitness
                    
                    visualizer.visualize_individual(0, individual, show_weights=True, precision=3)
                    
                except FileNotFoundError:
                    print("No saved weights found. Run evolution first.")
                    
            elif choice == '4':
                # Compare individuals
                print("Generating population for comparison...")
                pop = random_population()
                evaluate_fitness(pop)
                
                # Sort and show top individuals
                sorted_indices = np.argsort(pop[:, -1])
                print("\nTop 5 individuals:")
                for i in range(5):
                    idx = sorted_indices[i]
                    fitness = pop[idx, -1]
                    print(f"  {i}: Individual #{idx}, fitness = {fitness:.4f}")
                
                try:
                    indices_str = input("Enter individual indices to compare (e.g., '0,1'): ")
                    indices = [int(x.strip()) for x in indices_str.split(',')]
                    actual_indices = [sorted_indices[i] for i in indices]
                    visualizer.compare_individuals(pop, actual_indices)
                except (ValueError, IndexError):
                    print("Invalid indices.")
                    
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