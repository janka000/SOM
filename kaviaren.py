import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def differential_evolution_table_placement(Lx, Ly, r, n, max_generations=100, draw_after_each_iteration=False):
    # Initialization of the population
    population = np.random.uniform(low=r, high=(Lx - r, Ly - r), size=(n, 2))

    optimal_positions = []

    for generation in range(max_generations):
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

        if draw_after_each_iteration:
            # Visualization of results after each iteration
            plot_results(Lx, Ly, population, r)

        # Finding the best solution
        best_index = np.argmax([objective_function(ind, Lx, Ly, r) for ind in population])
        optimal_positions.append(tuple(population[best_index]))

    if not draw_after_each_iteration:
        # Visualization of final results
        plot_results(Lx, Ly, population, r)

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

def plot_results(Lx, Ly, positions, r):
    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    # Plotting tables
    for pos in positions:
        circle = Circle((pos[0], pos[1]), r, fill=False, color='blue')
        ax.add_artist(circle)

    # Plotting the cafe boundaries
    rectangle = plt.Rectangle((0, 0), Lx, Ly, fill=False, edgecolor='red', linewidth=2)
    ax.add_artist(rectangle)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Optimal Table Placement')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

if __name__ == "__main__":
    Lx = 30  # Width of the cafe
    Ly = 20  # Height of the cafe
    r = 1    # Radius of the table
    n = 5    # Number of tables

    optimal_positions = differential_evolution_table_placement(Lx, Ly, r, n, draw_after_each_iteration=True)
    print("Optimal positions:", optimal_positions)

