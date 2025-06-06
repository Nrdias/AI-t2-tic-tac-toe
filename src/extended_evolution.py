#!/usr/bin/env python3
"""
Extended evolution to see the process over many generations.
"""

from evolution_visualizer import *
import genetic

# Make it MUCH more challenging
genetic.TARGET_FITNESS = 0.05  # Extremely low target (near-perfect play required)
genetic.GAMES_PER_EVAL = 50    # Many more games for accurate evaluation
genetic.MUT_RATE = 0.05        # Lower mutation rate for more stable evolution
genetic.MUT_SIGMA = 0.1        # Smaller mutations

print("🚀 EXTENDED EVOLUTION MODE")
print("=" * 60)
print(f"Target Fitness: {genetic.TARGET_FITNESS} (extremely challenging!)")
print(f"Games per evaluation: {genetic.GAMES_PER_EVAL}")
print(f"Mutation rate: {genetic.MUT_RATE}")
print(f"Mutation sigma: {genetic.MUT_SIGMA}")
print()
print("This will require near-perfect play and show evolution over many generations!")
print("Press Ctrl+C to stop early if you want to see the intermediate results")
print()

# Run with very challenging parameters
visualizer = EvolutionVisualizer(num_tracked=20, max_gens=200)
best = visualizer.run_visual_evolution()

if best is not None:
    # Test the final evolved player
    best_weights = best[:-1]
    print("\n" + "="*60)
    print("🧪 TESTING THE HIGHLY EVOLVED PLAYER")
    print("="*60)
    
    from test_genetic import test_ai_vs_random, test_ai_vs_heuristic
    test_ai_vs_random(best_weights, 100)
    test_ai_vs_heuristic(best_weights, 100) 