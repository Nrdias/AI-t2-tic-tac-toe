#!/usr/bin/env python3
"""
Run evolution with more challenging parameters to see the process over multiple generations.
"""

from evolution_visualizer import *
import genetic

# Make it more challenging by requiring better performance
genetic.TARGET_FITNESS = 0.1  # Much lower target (better performance required)
genetic.GAMES_PER_EVAL = 20   # More games per evaluation for better accuracy

print("🔥 CHALLENGING EVOLUTION MODE")
print("=" * 50)
print(f"Target Fitness: {genetic.TARGET_FITNESS} (very challenging!)")
print(f"Games per evaluation: {genetic.GAMES_PER_EVAL}")
print("This will show evolution over multiple generations...")
print()

# Run with challenging parameters
visualizer = EvolutionVisualizer(num_tracked=20, max_gens=150)
best = visualizer.run_visual_evolution() 