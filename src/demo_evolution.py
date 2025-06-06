#!/usr/bin/env python3
"""
Demonstration of evolution over multiple generations with strict target.
"""

from evolution_visualizer import EvolutionVisualizer
from genetic import *
import numpy as np
import time

def demo_evolution():
    """
    Run evolution with parameters that guarantee multiple generations.
    """
    print("🎯 EVOLUTION DEMONSTRATION")
    print("=" * 60)
    print("This will show you exactly how 20 individuals evolve!")
    print("Target: Perfect play (fitness = 0.0)")
    print("=" * 60)
    
    # Create visualizer that tracks 20 individuals
    visualizer = EvolutionVisualizer(num_tracked=20, max_gens=100)
    
    # Initialize population
    pop = random_population()
    evaluate_fitness(pop)
    
    print("\n🚀 Starting evolution... Press Ctrl+C to stop\n")
    time.sleep(2)
    
    try:
        for gen in range(1, 101):  # Force 100 generations
            # Display current state
            visualizer.print_evolution_display(gen, pop)
            
            # Sort population by fitness  
            pop = pop[np.argsort(pop[:, -1])]
            
            # Check for perfect fitness (impossible target)
            best_fitness = pop[0, -1]
            if best_fitness <= 0.0:  # Perfect play
                break
                
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
            
            # Small delay to see the evolution
            time.sleep(0.5)  # Half second between generations
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Evolution stopped by user")
    
    # Final results
    visualizer.print_final_results(pop)
    return pop[np.argsort(pop[:, -1])][0]

if __name__ == "__main__":
    print("This demonstration will show 20 individuals evolving over generations.")
    print("You'll see their fitness bars change color and improve over time!")
    print("\nLegend:")
    print("🟢 Green bars = Excellent fitness (< 0.4)")
    print("🟡 Yellow bars = Good fitness (0.4 - 0.8)")  
    print("🔵 Blue bars = Poor fitness (0.8 - 1.2)")
    print("🔴 Red bars = Very poor fitness (> 1.2)")
    print("\nChange indicators:")
    print("↑ = Fitness improved (got better)")
    print("↓ = Fitness got worse") 
    print("←→ = No significant change")
    
    input("\nPress Enter to start the evolution demonstration...")
    
    best = demo_evolution()
    
    if best is not None:
        print(f"\n💾 Best evolved fitness: {best[-1]:.4f}")
        
        # Save and test
        np.save('demo_evolved_weights.npy', best[:-1])
        print("Weights saved to 'demo_evolved_weights.npy'")
        
        test_choice = input("\n🎮 Test the evolved player? (y/n): ")
        if test_choice.lower().startswith('y'):
            from test_genetic import test_ai_vs_random, test_ai_vs_heuristic
            test_ai_vs_random(best[:-1], 50)
            test_ai_vs_heuristic(best[:-1], 50) 