import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy

class Individual:
    def __init__(self, M, p, initial_positions, final_positions, random_final=True):
        self.M = M
        self.p = p
        self.initial_positions = initial_positions
        self.final_positions = random.sample(final_positions,len(final_positions)) if random_final else final_positions #random shuffle if new individual, else given
        self.fitness = self.evaluate()

    def evaluate(self):
        s = 0
        for i in range(self.M):
            s+=dist(self.initial_positions[i][0],self.initial_positions[i][1], self.final_positions[i][0], self.final_positions[i][1]) ** self.p  #summing distances between dron start positions and final positions (dron i starts at pos. initial_positions[i] and ends at final_positions[i])                      
        Lp = s ** (1/self.p) 
        #print("new Lp.. :",Lp)
                       
        self.fitness = - Lp # fitness is something we want to maximize, therefore, if we want to minimize the sum of distances, we have to use -Lp norm as fitness
        return self.fitness
        
    def mutate(self):
        # mutation by permuattion
        mutation_point1, mutation_point2 = random.sample(range(self.M), 2)
        self.final_positions[mutation_point1], self.final_positions[mutation_point2] = \
        self.final_positions[mutation_point2], self.final_positions[mutation_point1]
        self.fitness = self.evaluate() # important! recalculate fitness after mutation

def dist(p1x,p1y,p2x,p2y):
    return ((p2x-p1x)**2 + (p2y-p1y)**2)**(1/2)

def crossover(parent1, parent2):
    # single crossover
    crossover_point = random.randint(0, parent1.M - 1)
    child_positions = parent1.final_positions[:crossover_point] + [pos for pos in parent2.final_positions if pos not in parent1.final_positions[:crossover_point]]
    return Individual(parent1.M, parent1.p, copy.deepcopy(parent1.initial_positions), copy.deepcopy(child_positions), random_final=False)
    
def print_matching(individual):
    for i in range(individual.M):
        print("dron at position "+str(individual.initial_positions[i])+" goes to "+str(individual.final_positions[i]))

def plot_movement(ax, initial_positions, final_positions, best_positions, current_positions, current_iteration, current_fitness, best_iteration, best_fitness):
    ax.cla()

    # Plot the connections from initial positions to the best positions in dashed red lines
    for i in range(len(initial_positions)):
        x_values = [initial_positions[i][0], best_positions[i][0]]
        y_values = [initial_positions[i][1], best_positions[i][1]]
        ax.plot(x_values, y_values, linestyle='dashed', color='red', alpha=0.5, label='_nolegend_')  

    # Plot the connections from initial positions to the current positions in solid gray lines
    for i in range(len(initial_positions)):
        x_values = [initial_positions[i][0], current_positions[i][0]]
        y_values = [initial_positions[i][1], current_positions[i][1]]
        ax.plot(x_values, y_values, linestyle='-', color='gray', alpha=0.8, label='_nolegend_')  

    ax.scatter(*zip(*initial_positions), label='Init Positions', marker='o', s=100, color="#34aeeb")
    ax.scatter(*zip(*final_positions), label='Final Positions', marker='*', s=100, color="#ebd234")

    # Create a custom legend with "Best Solution," "Current Solution," "Start Points," and "End Points"
    custom_legend = [
        ax.plot([], [], linestyle='dashed', color='red', alpha=0.5)[0],
        ax.plot([], [], linestyle='-', color='gray', alpha=0.8)[0],
        ax.scatter([], [], label='Start Points', marker='o', s=100, color="#34aeeb"),
        ax.scatter([], [], label='End Points', marker='*', s=100, color="#ebd234")
    ]
    
    ax.legend(custom_legend, ['Best Solution', 'Current Solution', 'Start Points', 'End Points'])

    # display the iteration numbers and fitness
    ax.text(0.05, 0.05, f'Current Iteration: {current_iteration}', transform=ax.transAxes, color='gray', fontsize=10)
    ax.text(0.05, 0.10, f'Current Fitness: {current_fitness}', transform=ax.transAxes, color='gray', fontsize=10)
    ax.text(0.05, 0.15, f'Best Iteration: {best_iteration}', transform=ax.transAxes, color='red', fontsize=10)
    ax.text(0.05, 0.20, f'Best Fitness: {best_fitness}', transform=ax.transAxes, color='red', fontsize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Drones')


def genetic_algorithm(M, p, initial_positions, final_positions, population_size=100, generations=100, crossover_prob=0.2, mutation_prob=0.2, draw_after_each_it=False, print_after_each_it=False):
    population = [Individual(M, p, initial_positions, final_positions) for _ in range(population_size)]
    best_individual = max(population, key=lambda x: x.fitness) #maximize fitness
    current_best = max(population, key=lambda x: x.fitness) #maximize fitness
    best_positions = best_individual.final_positions
    best_generation = 0
    generation = 0
    if draw_after_each_it:
        fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        nonlocal population, best_individual, best_positions, best_generation, current_best, generation
        generation, _ = frame

        # crossover and mutations to generate offspring
        offspring = []
        parents = sorted(population, key=lambda x: -x.fitness)
        parents = population
        random.shuffle(parents)
        for p in range(population_size // 2):
            # select parents based on fitness
            parent1 = parents[p]
            parent2 = parents[p+1]

            if random.random() < crossover_prob:
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
            else:
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)

            if random.random() < mutation_prob:
                child1.mutate()
            if random.random() < mutation_prob:
                child2.mutate()

            offspring.extend([child1, child2])

        # replace old population with newer, based on fitness
        new_population = offspring + copy.deepcopy(parents)
        new_population.sort(key=lambda x: -x.fitness) #sort individuals by fitness (bigger fitness is better, therefore - in lambda expr.)
        population = new_population[:population_size] #survival of the fittest (take the best ones)
        #print(str([p.fitness for p in population]))

        current_best = max(population, key=lambda x: x.fitness) #maximize fitness
        current_positions = current_best.final_positions

        #update best solution if the current is better or equal 
        if current_best.fitness >= best_individual.fitness:
            best_individual = current_best
            best_positions = current_positions
            best_generation = generation

        if draw_after_each_it:
            plot_movement(ax, initial_positions, final_positions, best_positions, current_positions, generation, current_best.fitness, best_generation, best_individual.fitness)

    if draw_after_each_it:
        ani = FuncAnimation(plt.gcf(), update, frames=enumerate([None] * generations), repeat=False)
        plt.show()
        if print_after_each_it:
            print(f"Generation: {generation}, Current Fitness: {current_best.fitness}, Best Fitness: {best_individual.fitness}, Best generation: {best_generation}")
    elif print_after_each_it:
        for i in range(generations):
            update((i,None))
            print(f"Generation: {generation}, Current Fitness: {current_best.fitness}, Best Fitness: {best_individual.fitness}, Best generation: {best_generation}")
    else:
        for i in range(generations):
            update((i,None))

    best_positions = best_individual.final_positions

    print("\nFinal Solution:")
    print("Best Matching:")
    print_matching(best_individual)
    print("Best Fitness:", best_individual.fitness)

    # Draw the best solution at the end
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_movement(ax, initial_positions, final_positions, best_positions, best_positions, generations-1, current_best.fitness, best_generation, best_individual.fitness)
    plt.show()

class Test:
    def M5_simple_A():
       M = 5  # pocet dronov
       p = 1  # parameter p pre Lp-normu
       population_size = 100
       generations = 100
       crossover_prob = 0.2
       mutation_prob = 0.5   
    
       initial_positions = [(2,10),(2, 12),(2, 15),(2,17),(2,20)]
       F = [(4,10),(3, 12),(4, 15),(3,17),(4,20)]
       
       genetic_algorithm(M, p, initial_positions, F, population_size=population_size, generations=generations, crossover_prob=crossover_prob, mutation_prob=mutation_prob, draw_after_each_it=False, print_after_each_it=False)
   
    def M5_simple_B():
       M = 5  # pocet dronov
       p = 1  # parameter p pre Lp-normu
       population_size = 100
       generations = 100
       crossover_prob = 0.2
       mutation_prob = 0.5   
    
       initial_positions = [(10,2),(12,2),(15,2),(17,2),(20,2)]
       F = [(10,3),(12,4),(15,5),(17,6),(20,7)]
       
       genetic_algorithm(M, p, initial_positions, F, population_size=population_size, generations=generations, crossover_prob=crossover_prob, mutation_prob=mutation_prob, draw_after_each_it=False, print_after_each_it=False)
   
    def M10_simple_A():
       M = 10  # pocet dronov
       p = 1  # parameter p pre Lp-normu
       population_size = 100
       generations = 100
       crossover_prob = 0.2
       mutation_prob = 0.5   
    
       initial_positions = [(2,10),(2, 12),(2, 15),(2,17),(2,20),(2,22),(2, 24),(2, 26),(2,28),(2,31)]
       F = [(4,10),(3, 12),(4, 15),(3,17),(4,20),(5,22),(3, 24),(4, 26),(3,28),(4,31)] 
       
       genetic_algorithm(M, p, initial_positions, F, population_size=population_size, generations=generations, crossover_prob=crossover_prob, mutation_prob=mutation_prob, draw_after_each_it=False, print_after_each_it=False)
   
    def M10_simple_B():
        M = 10  # pocet dronov
        p = 1  # parameter p pre Lp-normu
        population_size = 100
        generations = 100
        crossover_prob = 0.2
        mutation_prob = 0.5   
    
        initial_positions = [(10,2),(12,2),(15,2),(17,2),(20,2),(23,2),(25,2),(27,2),(29,2),(30,2)]
        F = [(10,3),(12,4),(15,5),(17,6),(20,7),(23,3),(25,4),(27,5),(29,10),(30,9)]
        
        genetic_algorithm(M, p, initial_positions, F, population_size=population_size, generations=generations, crossover_prob=crossover_prob, mutation_prob=mutation_prob, draw_after_each_it=False, print_after_each_it=False)
   
    def random_10x10(M_drons, animated = False):
        M = M_drons  # pocet dronov
        p = 1  # parameter p pre Lp-normu
        population_size = M*10
        generations = 100
        crossover_prob = 0.2
        mutation_prob = 0.5   
    
        #generate random initial and end positions
        initial_positions = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(M)]
        F = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(M)] 
        
        genetic_algorithm(M, p, initial_positions, F, population_size=population_size, generations=generations, crossover_prob=crossover_prob, mutation_prob=mutation_prob, draw_after_each_it=animated, print_after_each_it=False)


if __name__ == "__main__":

    
    Test.M5_simple_A()
    Test.M5_simple_B()
    
    Test.M10_simple_A()
    Test.M10_simple_B()
    
    Test.random_10x10(10)
    Test.random_10x10(20)
    
    Test.random_10x10(15, animated = True)

