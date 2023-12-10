import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

def differential_evolution_table_placement(Lx, Ly, r, n, max_generations=100, draw_after_each_iteration=False):
    # Initialization of the population
    population = np.random.uniform(low=r, high=(Lx - r, Ly - r), size=(n, 2))

    optimal_positions = []

    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    def update(frame):
        nonlocal population

        for i in range(n):
            # Mutation
            indices = [idx for idx in range(n) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = population[a] + 0.5 * (population[b] - population[c])

            # Crossover
            crossover_prob = np.random.rand()
            trial = np.where(np.random.rand(2) < crossover_prob, mutant, population[i])

            # Selection
            if objective_function(trial, Lx, Ly, r) > objective_function(population[i], Lx, Ly, r) and constraints(trial, Lx, Ly, r):
                population[i] = trial

        # Visualization of results after each iteration
        plot_results(ax, Lx, Ly, population, r)

        # Finding the best solution
        best_index = np.argmax([objective_function(ind, Lx, Ly, r) for ind in population])
        optimal_positions.append(tuple(population[best_index]))

    anim = FuncAnimation(fig, update, frames=max_generations, repeat=False)
    plt.show()

    return optimal_positions

def objective_function(positions, Lx, Ly, r):
    # Objective function - negative of the minimum distance between table centers
    min_distance = float('inf')

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[j]) - 2 * r
            min_distance = min(min_distance, distance)

    return -min_distance

def constraints(positions, Lx, Ly, r):
    # Constraints for the distance of the table from the boundaries of the cafe
    if isinstance(positions, list):
        # For a list of positions
        for pos in positions:
            if pos[0] - r < 0 or pos[0] + r > Lx or pos[1] - r < 0 or pos[1] + r > Ly:
                return False
    else:
        # For a single position (tuple)
        if positions[0] - r < 0 or positions[0] + r > Lx or positions[1] - r < 0 or positions[1] + r > Ly:
            return False

    return True

def plot_results(ax, Lx, Ly, positions, r):
    ax.clear()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    # Plotting tables
    for pos in positions:
        circle = Circle((pos[0], pos[1]), r, fill=False, color='blue')
        ax.add_artist(circle)

    # Plotting the cafe boundaries
    rectangle = plt.Rectangle((0, 0), Lx, Ly, fill=False, edgecolor='red', linewidth=2)
    ax.add_artist(rectangle)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Optimal Table Placement')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

if __name__ == "__main__":
    Lx = 30  # Width of the cafe
    Ly = 20  # Height of the cafe
    r = 1    # Radius of the table
    n = 5    # Number of tables

    optimal_positions = differential_evolution_table_placement(Lx, Ly, r, n, draw_after_each_iteration=True)
    print("Optimal positions:", optimal_positions)

