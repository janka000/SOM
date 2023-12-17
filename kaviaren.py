import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import copy


class InitialPopulation:

    def random(Lx, Ly, r, n, population_size, draw_init=False, print_check=False):
        def draw(p, color):
            fig, ax = plt.subplots()
            ax.set_xlim(0, Lx)
            ax.set_ylim(0, Ly)
            ax.clear()
            plot_results(ax, Lx, Ly, p, r, n, 0, 0, 0, color=color)
            plt.show()
        # initialization of the population (individual == n tables in cafe)
        population = [[] for i in range(population_size)]
        # generate population and ensure that each individual is feasibe
        for i in range(population_size):
            while True: #while inital solution violates constraints, generate new one
                population[i] = np.random.uniform(low=(r,r), high=(Lx-r, Ly-r), size=(n,2)) #distribute the tables randomly
                if constraints(population[i], Lx, Ly, r,n): #check if new solution violates constarints
                    break
            if draw_init:
                print(population[i])
                draw(population[i], "green")
                
        if print_check:
            InitialPopulation.check(population,population_size)
        
        return population 


    def grid(Lx, Ly, r, n, population_size, print_check=False):
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
        
        if print_check:
            InitialPopulation.check(population,population_size)   
                
        return population
        
        def check(population, poppulation_size):
            for i in range(population_size):
                print("population ", i, ":: is feasible solution?: ", constraints(population[i],Lx,Ly,r,n))



def differential_evolution_table_placement(Lx, Ly, r, n, F = 0.7, crossover_prob = 0.5, population_size=1000, max_generations=1000, draw_after_each_it=False, init="random"):
    
    if init=="random":
        population = InitialPopulation.random(Lx, Ly, r, n, population_size, draw_init=False, print_check=False)
    elif init=="grid":
        population = InitialPopulation.grid(Lx, Ly, r, n, population_size, print_check=False)
    else:
        print("unknown init startegy")
        exit(1)
    
    print("finished initialization")
    generation = 0
    best_generation = 0
    
    # find the best solution in the init. population
    best_index = np.argmax([objective_function(ind, Lx, Ly, r, n) for ind in population])
    best_solution = population[best_index]
    print("best solution in init. population:", best_index, "is feasible?:", constraints(best_solution, Lx, Ly, r, n))
    
    
    #drawing stuff
    if draw_after_each_it:
        fig, ax = plt.subplots()
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)

    def update(frame):
        nonlocal population, best_solution, best_index, generation, best_generation
         
        generation+=1

        for i in range(population_size):
            # mutation
            mutant = mutate(population, population_size, i, F, Lx, Ly, r)

            # crossover
            trial = crossover(mutant, population[i], crossover_prob, Lx, Ly, r, n)

            # selection 
            """
            Replace xi with z if f(z)<=f(xi); in the opposite case, discard z as a failed attempt to improve the position of xi
            """
            trial_objective = objective_function(trial, Lx, Ly, r, n)
            current_objective = objective_function(population[i], Lx, Ly, r, n)

            if trial_objective > current_objective and constraints(trial, Lx, Ly, r, n): #comapre objective function values and check the constarints (if trial max > current. max, update)
                population[i] = trial
                
                # update best_index, best_solution, and best_genaration if the new is better
                if trial_objective > objective_function(best_solution, Lx, Ly, r, n):
                    best_index = i
                    best_solution = trial
                    best_generation = generation
                    print("new best solution found in: generation ", generation, "individual", i)
                
        if draw_after_each_it:
            # visualization of results after each iteration
            plot_results(ax, Lx, Ly, best_solution, r, n, generation, best_index, best_generation)

    if draw_after_each_it:
        animation = FuncAnimation(fig, update, frames=max_generations-1, repeat=False)
        plt.show()
    else:
        for i in range(max_generations):
            update((i,None))



    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    plot_results(ax, Lx, Ly, best_solution, r, n, generation, best_index, best_generation)
    plt.show()

    return best_solution


def crossover(mutant, parent, crossover_prob, Lx, Ly, r, n):
    """
    compute a challenger z (== all tables placement in cafe) via the uniform crossover of xi and y, that is, 
    each component of z (==table) is independently inherited from xi with probability 1-CR and from y with probability CR. 
    """
    # perform crossover to generate a trial solution
    trial = copy.deepcopy(parent) #set trial to copy of parent
    mutant = copy.deepcopy(mutant)
    
    crossover_mask = np.random.rand(n) < crossover_prob #random (uniform distribution) T/F with given prob. for each table 

    trial[crossover_mask, 0] = mutant[crossover_mask, 0] #replace x coordinate of the table with mutant if T
    trial[:, 0] = np.clip(trial[:, 0], r, Lx - r)  # clip to ensure we are within bounds

    trial[crossover_mask, 1] = mutant[crossover_mask, 1] #replace y coordinate of table with mutant if T
    trial[:, 1] = np.clip(trial[:, 1], r, Ly - r)  # clip to ensure we are within bounds

    return trial

def mutate(population, population_size, i, F, Lx, Ly, r):
    """
    randomly generate a base vector a and two shift vectors b, c. The vectors a, b, c must all be 
    members of the current population and all four vectors xi, a, b, c, must be distinct. 
    Use a, b, c to form the donor vector... 
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

def objective_function(positions, Lx, Ly, r, n):
    # objective function - in our case something we want to MAXIMIZE (we want to maximize minimal distance), 
    # so the objective function will be minimal distance between the tables 
    min_distance = float('inf')

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(positions[i] - positions[j]) - 2 * r
            min_distance = min(min_distance, distance)

    return min_distance

def constraints(positions, Lx, Ly, r, n):
    for j in range(n):
        # constraints for the distance of each table from the boundaries of the cafe
        # i.e. the table can not escape from the cafe
        if positions[j, 0] - r < 0 or positions[j, 0] + r > Lx or positions[j, 1] - r < 0 or positions[j, 1] + r > Ly:
            return False
        # constarins for each table not to intersect with another 
        # (no stacked tables, thant would be impossible or suboptimal in covid times)
        for k in range(j + 1, n):
            distance = np.linalg.norm(positions[j] - positions[k])
            if distance < 2 * r:
                return False
    return True

def plot_results(ax, Lx, Ly, positions, r, n, generations, best_index, best_generation, color="brown"):
    ax.clear()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    # plotting the cafe floor, because why not
    rectangle = plt.Rectangle((0, 0), Lx, Ly, fill=True, color='gray', linewidth=2)
    ax.add_artist(rectangle)

    # plotting tables
    for i in range(n):
        circle = Circle((positions[i][0], positions[i][1]), r, fill=True, color=color)
        ax.add_artist(circle)
        
    ax.text(0.05, 0.05, f'Generation: {generations}', transform=ax.transAxes, color='red', fontsize=10)
    ax.text(0.05, 0.10, f'Best index: {best_index}', transform=ax.transAxes, color='red', fontsize=10)
    ax.text(0.05, 0.15, f'Best generation: {best_generation}', transform=ax.transAxes, color='red', fontsize=10)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Optimal Table Placement')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

class Test:
    def rectangle_4(animate=True):
        Lx = 30  # width of cafe
        Ly = 20  # height of cafe
        r = 3    # table radius
        n = 4    # num. of tables
        population_size = 50  # num. of individuals in the population
        max_generations = 200
        F_param = 0.9 # mutation param.
        crossover_prob = 0.8

        optimal_positions = differential_evolution_table_placement(Lx, Ly, r, n, population_size=population_size, max_generations=max_generations, F= F_param, crossover_prob=crossover_prob, draw_after_each_it=animate)
        print("Optimal positions:", optimal_positions)

    def square_8(animate=True):
        Lx = 6
        Ly = 6
        r = 0.5
        n = 8
        population_size = 50
        max_generations = 1500
        F_param = 0.8
        crossover_prob = 0.9

        optimal_positions = differential_evolution_table_placement(Lx, Ly, r, n, population_size=population_size, max_generations=max_generations, F= F_param, crossover_prob=crossover_prob, draw_after_each_it=animate)
        print("Optimal positions:", optimal_positions)

    def square_5(animate=True):
        Lx = 6
        Ly = 6
        r = 0.5
        n = 5
        population_size = 50
        max_generations = 200
        F_param = 0.8
        crossover_prob = 0.9

        optimal_positions = differential_evolution_table_placement(Lx, Ly, r, n, population_size=population_size, max_generations=max_generations, F= F_param, crossover_prob=crossover_prob, draw_after_each_it=animate)
        print("Optimal positions:", optimal_positions)
        
    def square_7(animate=True):
        Lx = 6
        Ly = 6
        r = 0.5
        n = 7
        population_size = 50
        max_generations = 1000
        F_param = 0.8
        crossover_prob = 0.9

        optimal_positions = differential_evolution_table_placement(Lx, Ly, r, n, population_size=population_size, max_generations=max_generations, F= F_param, crossover_prob=crossover_prob, draw_after_each_it=animate)
        print("Optimal positions:", optimal_positions)


if __name__ == "__main__":

    Test.rectangle_4(animate=False)
    Test.square_5(animate=False)
    Test.square_8(animate=False)
