#!/usr/bin/env python3
"""
Evolution Visualizer for Genetic Algorithm Tic-Tac-Toe

This script creates a visual representation of how 20 individuals evolve
over generations, showing their fitness progression in real-time.
"""

import numpy as np
from genetic import *
import time
import os
import sys

class EvolutionVisualizer:
    def __init__(self, num_tracked=20, max_gens=200):
        self.num_tracked = num_tracked
        self.max_gens = max_gens
        self.tracked_fitness = []  # List of arrays, each containing fitness of tracked individuals
        self.generation_data = []
        
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def create_fitness_bar(self, fitness, max_fitness=2.0, width=30):
        """Create a visual bar representation of fitness."""
        # Invert fitness since lower is better (0 = best, 2 = worst)
        normalized = 1.0 - (fitness / max_fitness)
        filled_length = int(width * normalized)
        
        # Create color-coded bar
        if fitness <= 0.4:
            color_code = '\033[92m'  # Green for good fitness
        elif fitness <= 0.8:
            color_code = '\033[93m'  # Yellow for medium fitness  
        elif fitness <= 1.2:
            color_code = '\033[94m'  # Blue for poor fitness
        else:
            color_code = '\033[91m'  # Red for very poor fitness
        
        reset_code = '\033[0m'
        
        bar = '█' * filled_length + '░' * (width - filled_length)
        return f"{color_code}{bar}{reset_code}"
    
    def print_evolution_display(self, generation, pop):
        """Print the current evolution state."""
        self.clear_screen()
        
        # Sort population by fitness
        sorted_pop = pop[np.argsort(pop[:, -1])]
        
        # Track the top individuals
        current_fitness = sorted_pop[:self.num_tracked, -1]
        self.tracked_fitness.append(current_fitness.copy())
        
        # Store generation statistics
        best_fitness = sorted_pop[0, -1]
        avg_fitness = np.mean(sorted_pop[:, -1])
        worst_fitness = sorted_pop[-1, -1]
        
        self.generation_data.append({
            'gen': generation,
            'best': best_fitness,
            'avg': avg_fitness,
            'worst': worst_fitness
        })
        
        # Print header
        print("=" * 80)
        print(f"🧬 GENETIC ALGORITHM EVOLUTION - GENERATION {generation}")
        print("=" * 80)
        print(f"Population Size: {POP_SIZE} | Elite Size: {ELITE_SIZE} | Target: {TARGET_FITNESS}")
        print("-" * 80)
        
        # Print population statistics
        print(f"📊 GENERATION {generation} STATISTICS:")
        print(f"   Best Fitness:  {best_fitness:.4f} {self.create_fitness_bar(best_fitness, width=20)}")
        print(f"   Avg Fitness:   {avg_fitness:.4f} {self.create_fitness_bar(avg_fitness, width=20)}")
        print(f"   Worst Fitness: {worst_fitness:.4f} {self.create_fitness_bar(worst_fitness, width=20)}")
        print("-" * 80)
        
        # Print top 20 individuals
        print(f"🏆 TOP {self.num_tracked} INDIVIDUALS:")
        print("Rank | Fitness  | Progress Bar                    | Change")
        print("-" * 65)
        
        for i in range(self.num_tracked):
            fitness = current_fitness[i]
            bar = self.create_fitness_bar(fitness)
            
            # Calculate change from previous generation
            change_str = "    NEW"
            if len(self.tracked_fitness) > 1:
                prev_fitness = self.tracked_fitness[-2][i] if i < len(self.tracked_fitness[-2]) else float('inf')
                change = fitness - prev_fitness
                if abs(change) < 0.001:
                    change_str = "   ←→"
                elif change < 0:
                    change_str = f"  ↑{abs(change):.3f}"  # Improvement (lower is better)
                else:
                    change_str = f"  ↓{change:.3f}"       # Deterioration
            
            print(f" {i+1:2d}  | {fitness:.4f}  | {bar} | {change_str}")
        
        print("-" * 80)
        
        # Print evolution trend (last 10 generations)
        if len(self.generation_data) >= 2:
            print("📈 EVOLUTION TREND (Last 10 Generations):")
            start_idx = max(0, len(self.generation_data) - 10)
            trend_data = self.generation_data[start_idx:]
            
            print("Gen:  ", end="")
            for data in trend_data:
                print(f"{data['gen']:4d}", end=" ")
            print()
            
            print("Best: ", end="")
            for data in trend_data:
                print(f"{data['best']:4.2f}", end=" ")
            print()
            
            print("Avg:  ", end="")
            for data in trend_data:
                print(f"{data['avg']:4.2f}", end=" ")
            print()
        
        print("-" * 80)
        
        # Print progress indicators
        progress = generation / self.max_gens
        progress_bar = int(50 * progress)
        progress_display = '█' * progress_bar + '░' * (50 - progress_bar)
        
        print(f"⏱️  Progress: {progress:.1%} [{progress_display}] Gen {generation}/{self.max_gens}")
        
        if best_fitness <= TARGET_FITNESS:
            print(f"🎯 TARGET ACHIEVED! Best fitness {best_fitness:.4f} ≤ {TARGET_FITNESS}")
        
        print("=" * 80)
        
        # Brief pause to make it readable
        time.sleep(0.1)
    
    def run_visual_evolution(self):
        """Run the genetic algorithm with visual tracking."""
        print("🚀 Starting Visual Evolution...")
        print("Press Ctrl+C to stop early")
        time.sleep(2)
        
        # Initialize population
        pop = random_population()
        evaluate_fitness(pop)
        
        try:
            for gen in range(1, self.max_gens + 1):
                # Display current state
                self.print_evolution_display(gen, pop)
                
                # Check if target reached
                best_fitness = np.min(pop[:, -1])
                if best_fitness <= TARGET_FITNESS:
                    break
                
                # Evolution step
                pop = pop[np.argsort(pop[:, -1])]  # Sort by fitness
                
                # Create next generation
                new_pop = np.empty_like(pop)
                new_pop[:ELITE_SIZE] = pop[:ELITE_SIZE].copy()  # Elitism
                
                # Crossover
                crossover(pop, new_pop)
                
                # Mutation
                mutate(new_pop)
                
                # Mark fitness unknown for non-elites
                new_pop[ELITE_SIZE:, -1] = np.inf
                pop = new_pop
                
                # Evaluate fitness
                evaluate_fitness(pop)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Evolution stopped by user")
        
        # Final results
        self.print_final_results(pop)
        
        return pop[np.argsort(pop[:, -1])][0]  # Return best individual
    
    def print_final_results(self, pop):
        """Print final evolution results."""
        print("\n" + "=" * 80)
        print("🏁 EVOLUTION COMPLETED!")
        print("=" * 80)
        
        sorted_pop = pop[np.argsort(pop[:, -1])]
        best_fitness = sorted_pop[0, -1]
        
        print(f"🏆 Best Individual Fitness: {best_fitness:.4f}")
        print(f"🎯 Target Fitness: {TARGET_FITNESS}")
        
        if best_fitness <= TARGET_FITNESS:
            print("✅ TARGET ACHIEVED!")
        else:
            print(f"📊 Progress: {((2.0 - best_fitness) / 2.0) * 100:.1f}% toward perfect play")
        
        print(f"📈 Total Generations: {len(self.generation_data)}")
        
        if len(self.generation_data) >= 2:
            improvement = self.generation_data[0]['best'] - self.generation_data[-1]['best']
            print(f"📉 Total Improvement: {improvement:.4f}")
        
        print("=" * 80)

def main():
    """Main function with options."""
    print("🧬 Genetic Algorithm Evolution Visualizer")
    print("=" * 50)
    print("1. Quick Evolution (50 generations)")
    print("2. Standard Evolution (200 generations)")
    print("3. Full Evolution (600 generations)")
    print("4. Custom Evolution")
    print("5. Exit")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            visualizer = EvolutionVisualizer(num_tracked=20, max_gens=50)
            # Reduce games per evaluation for speed
            global GAMES_PER_EVAL
            original_games = GAMES_PER_EVAL
            GAMES_PER_EVAL = 10
            try:
                best = visualizer.run_visual_evolution()
                return best
            finally:
                GAMES_PER_EVAL = original_games
                
        elif choice == '2':
            visualizer = EvolutionVisualizer(num_tracked=20, max_gens=200)
            return visualizer.run_visual_evolution()
            
        elif choice == '3':
            visualizer = EvolutionVisualizer(num_tracked=20, max_gens=600)
            return visualizer.run_visual_evolution()
            
        elif choice == '4':
            try:
                gens = int(input("Number of generations: "))
                tracked = int(input("Number of individuals to track (default 20): ") or "20")
                visualizer = EvolutionVisualizer(num_tracked=tracked, max_gens=gens)
                return visualizer.run_visual_evolution()
            except ValueError:
                print("Invalid input. Using defaults.")
                return main()
                
        elif choice == '5':
            print("Goodbye!")
            return None
            
        else:
            print("Invalid choice. Please try again.")
            return main()
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return None

if __name__ == "__main__":
    best_individual = main()
    
    if best_individual is not None:
        # Save the best individual
        best_weights = best_individual[:-1]  # Exclude fitness
        np.save('evolved_weights.npy', best_weights)
        print(f"\n💾 Best weights saved to 'evolved_weights.npy'")
        
        # Offer to test the evolved player
        test_choice = input("\n🎮 Would you like to test the evolved player? (y/n): ")
        if test_choice.lower().startswith('y'):
            from test_genetic import test_ai_vs_random, test_ai_vs_heuristic
            print("\n🧪 Testing evolved player...")
            test_ai_vs_random(best_weights, 50)
            test_ai_vs_heuristic(best_weights, 50) 