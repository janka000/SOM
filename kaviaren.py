import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import copy
import time

def differential_evolution_table_placement(Lx, Ly, r, n, F = 0.7, crossover_prob = 0.5, population_size=1000, max_generations=1000, draw_after_each_iteration=False):
    
    #population = initial_population_squares(Lx, Ly, r, n, population_size, draw_init=False, print_check=False)
    population = initial_population_random(Lx, Ly, r, n, population_size, draw_init=False, print_check=False)
    
    print("finished initialization")
    generations = 0
    best_generation = 0
    
    # Find the best solution in the init. population
    best_index = np.argmax([objective_function(ind, Lx, Ly, r) for ind in population])
    best_solution = population[best_index]
    print("best solution in init. population:", best_index, "is feasible?:", constraints(best_solution, Lx, Ly, r))
    
    
    #drawing stuff
    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    def update(frame):
        nonlocal population, best_solution, best_index, generations, best_generation
         
        generations+=1

        for i in range(population_size):
            # Mutation
            mutant = mutate(population, i, F, Lx, Ly, r)

            # Crossover
            trial = crossover(mutant, population[i], crossover_prob, Lx, Ly, r)

            # Selection 
            """
            Replace
xi with z if f(z)<=f(xi); in the opposite case, discard z as a failed attempt to improve the position of xi
            """
            trial_objective = objective_function(trial, Lx, Ly, r)
            current_objective = objective_function(population[i], Lx, Ly, r)
            
            #if constraints(trial, Lx, Ly, r):
            #    print("trial objective", trial_objective, "current objective", current_objective)

            if trial_objective > current_objective and constraints(trial, Lx, Ly, r): #comapre objective function values and check the constarints (if trial max > current. max, update)
                population[i] = trial
                
                # Update best_index and best_solution if the new is better
                if trial_objective > objective_function(best_solution, Lx, Ly, r):
                    best_index = i
                    best_solution = trial
                    best_generation = generations
                    print("new best solution found in: generation ", generations, "iteration", i)
                
        
        # Visualization of results after each iteration
        plot_results(ax, Lx, Ly, best_solution, r, generations, best_index, best_generation)

    anim = FuncAnimation(fig, update, frames=max_generations-1, repeat=False)
    plt.show()


    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    plot_results(ax, Lx, Ly, best_solution, r, generations, best_index, best_generation)
    plt.show()

    return best_solution


def initial_population_random(Lx, Ly, r, n, population_size, draw_init=False, print_check=False):
    def draw(p, color):
        fig, ax = plt.subplots()
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.clear()
        plot_results(ax, Lx, Ly, p, r, 0, 0, 0, color=color)
        plt.show()
    # Initialization of the population
    population = np.random.uniform(low=(r,r), high=(Lx - r, Ly - r), size=(population_size, n, 2))
    #ensure that initial positions are feasibe
    for i in range(population_size):
        if draw_init:
            print("original placement:")
            print(population[i])
        while True: #inital solution violates constraints
            if draw_init:
                print("is wrong placement:")
                print(population[i])
                draw(population[i], "red")
            population[i] = np.random.uniform(low=(r,r), high=(Lx-r, Ly-r), size=(n,2))
            if constraints(population[i], Lx, Ly, r):
                break
        if draw_init:
            print("correct:")
            print(population[i])
            draw(population[i], "green")
        #print("feasible? ",i, constraints(population[i], Lx, Ly, r))
            
    if print_check:
        for i in range(population_size):
            print("population ", i, ":: is feasible solution?: ", constraints(population[i],Lx,Ly,r))
    
    return population 


def initial_population_squares(Lx, Ly, r, n, population_size, draw_init=False, print_check=False):
    population = []
    num_of_x_squares = int((Lx-2*r)*1000)//int((2*r)*1000)
    num_of_y_squares = int((Ly-2*r)*1000)//int((2*r)*1000)
    available = [i for i in range(num_of_x_squares*num_of_y_squares)]
    for i in range(population_size):
        placement = []
        positions = np.random.choice(available, size=n, replace=False) # n positions for n circles
        for c in positions:
            #calculate circle position
            c_x = r + (c%num_of_x_squares)*2*r
            c_y = r + (c//num_of_x_squares)*2*r
            placement.append([c_x,c_y])
        population.append(placement) 
    population = np.array(population)
    
    #if print_check:
    #    for i in range(population_size):
    #        print("population ", i, ":: is feasible solution?: ", constraints(population[i],Lx,Ly,r))        
            
    return population
        
     



def crossover(mutant, parent, crossover_prob, Lx, Ly, r):
    """
    compute a challenger z (== all tables placement in cafe) via the uniform crossover of xi and y, that is, each component of z (==table) is independently inherited
from xi with probability 1-CR and from y with probability CR. 
    """
    # Perform crossover to generate a trial solution
    trial = copy.deepcopy(parent) #set trial to copy of parent
    mutant = copy.deepcopy(mutant)
    
    crossover_mask = np.random.rand(n) < crossover_prob #random (uniform distribution) T/F with given prob. for each table 

    trial[crossover_mask, 0] = mutant[crossover_mask, 0] #replace x coordinate of the table with mutant if T
    trial[:, 0] = np.clip(trial[:, 0], r, Lx - r)  # clip to ensure we are within bounds

    trial[crossover_mask, 1] = mutant[crossover_mask, 1] #replace y coordinate of table with mutant if T
    trial[:, 1] = np.clip(trial[:, 1], r, Ly - r)  # clip to ensure we are within bounds

    return trial

def mutate(population, i, F, Lx, Ly, r):
    """
    randomly generate a base vector a and two shift vectors b, c. The vectors a, b, c must all be members of the current population and all four vectors xi, a, b, c, must be distinct. Use a, b, c to form the donor vector... 
    """
    indices = np.delete(np.arange(population_size), i)
    np.random.shuffle(indices)
    a, b, c = indices[:3]
    #print("mutant indices", a,b,c)
    mutant = population[a] + F * (population[b] - population[c])

    # (partialy) ensure feasibility by checking rectangle constraints for each table and clip them if the table is away..
    mutant[:, 0] = np.clip(mutant[:, 0], r, Lx - r)
    mutant[:, 1] = np.clip(mutant[:, 1], r, Ly - r)

    return mutant

def objective_function(positions, Lx, Ly, r):
    # Objective function - negative of the minimum distance between table centers
    min_distance = float('inf')

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(positions[i] - positions[j]) - 2 * r
            min_distance = min(min_distance, distance)

    return min_distance

def constraints(positions, Lx, Ly, r):
    #print("constraint check",positions, Lx, Ly, r)
    for j in range(n):
        # constraints for the distance of each table from the boundaries of the cafe
        if positions[j, 0] - r < 0 or positions[j, 0] + r > Lx or positions[j, 1] - r < 0 or positions[j, 1] + r > Ly:
            return False
        #constarins for each table not to intersect with another
        for k in range(j + 1, n):
            distance = np.linalg.norm(positions[j] - positions[k])
            if distance < 2 * r:
                return False
    return True

def plot_results(ax, Lx, Ly, positions, r, generations, best_index, best_generation, color="brown"):
    ax.clear()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

     # Plotting the cafe floor
    rectangle = plt.Rectangle((0, 0), Lx, Ly, fill=True, color='gray', linewidth=2)
    ax.add_artist(rectangle)

    # Plotting tables
    for i in range(n):
        circle = Circle((positions[i][0], positions[i][1]), r, fill=True, color=color)
        ax.add_artist(circle)
        
    ax.text(0.05, 0.05, f'Generation: {generations}', transform=ax.transAxes, color='red', fontsize=10)
    ax.text(0.05, 0.10, f'Best index: {best_index}', transform=ax.transAxes, color='red', fontsize=10)
    ax.text(0.05, 0.15, f'Best generation: {best_generation}', transform=ax.transAxes, color='red', fontsize=10)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Optimal Table Placement')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

if __name__ == "__main__":

    """
    Lx = 30  # Width of the cafe
    Ly = 20  # Height of the cafe
    r = 3    # Radius of the table
    n = 4    # Number of tables
    population_size = 10  # Number of individuals in the population
    max_generations = 1000
    F = 0.9
    crossover_prob = 0.8
    """

    
    Lx = 6
    Ly = 6
    r = 0.5
    n = 8
    population_size = 20
    max_generations = 1000
    F = 0.8
    crossover_prob = 0.9
    

    optimal_positions = differential_evolution_table_placement(Lx, Ly, r, n, population_size=population_size, max_generations=max_generations, F=F, crossover_prob=crossover_prob, draw_after_each_iteration=True)
    print("Optimal positions:", optimal_positions)

