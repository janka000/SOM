import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import copy

def differential_evolution_table_placement(Lx, Ly, r, n, max_generations=100, draw_after_each_iteration=False):
    # Initialization of the population ..of size 1
    population = np.random.uniform(low=r, high=(Lx - r, Ly - r), size=(n, 2))

    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    def update(frame):
        nonlocal population

        for i in range(n):
            # Mutation
            mutant = mutate(population, i, 0.5, Lx, Ly, r)

            # Crossover
            trial = crossover(mutant, population[i], 0.5, Lx, Ly, r)


            trial_population = copy.deepcopy(population)
            trial_population[i] = trial

            # Selection (check)
            if objective_function(trial_population, Lx, Ly, r) > objective_function(population, Lx, Ly, r) and constraints(trial_population, Lx, Ly, r):
                population[i] = trial

        print(population)
        # Visualization of results after each iteration
        plot_results(ax, Lx, Ly, population, r)


    anim = FuncAnimation(fig, update, frames=max_generations, repeat=False)
    plt.show()

    return population

def crossover(mutant, parent, crossover_prob, Lx, Ly, r): #crossover between mutant and parent in one tables placement
    # Perform crossover to generate a trial solution
    trial = np.copy(parent)
    crossover_mask = np.random.rand(2) < crossover_prob

    # Ensure that the trial solution respects boundaries
    trial[0] = mutant[0] if crossover_mask[0] else parent[0]
    trial[0] = max(r, min(trial[0], Lx - r))  # Clip to ensure within bounds
    
    trial[1] = mutant[1] if crossover_mask[1] else parent[1]
    trial[1] = max(r, min(trial[1], Ly - r))  # Clip to ensure within bounds

    return trial

def mutate(population, i, F, Lx, Ly, r): #move one table in one tables placement (table i)
    # Mutation using DE/rand/1 strategy
    #indices = [idx for idx in range(len(population)) if idx != i]
    #a, b, c = np.random.choice(indices, 3, replace=False)
    #mutant = population[a] + F * (population[b] - population[c])
    
    # Mutation using DE/best/1 strategy
    a, b, c = np.argsort([objective_function(ind, Lx, Ly, r) for ind in population])[-3:]
    best_individual = population[a]
    mutant = best_individual + F * (population[b] - population[c])
    
    #respect boundaries - move to the closest feasible place if not
    mutant[0]=max(r, min(mutant[0], Lx - r))
    mutant[1]=max(r, min(mutant[1], Ly - r))
    
    
    return mutant

def objective_function(positions, Lx, Ly, r):
    # Objective function - negative of the minimum distance between table centers
    min_distance = float('inf')

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[j]) - 2 * r
            min_distance = min(min_distance, distance)

    return -min_distance

def constraints(positions, Lx, Ly, r):

    #check all tables are withing cafe rectangle
    for i in range(len(positions)):
        if positions[i][0] - r < 0 or positions[i][0] + r > Lx or positions[i][1] - r < 0 or positions[i][1] + r > Ly:
            return False
    #check no tables intersect
    for j in range(len(positions)):
        if j != i:  # Skip checking the circle against itself
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < r + r:
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

