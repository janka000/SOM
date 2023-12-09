import numpy as np
import matplotlib.pyplot as plt

def differential_evolution(objective_function, bounds, constraints, population_size=10, crossover_prob=0.8, max_generations=100, draw_after_each_iteration=False):
    # Inicializácia populácie
    population = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(population_size, len(bounds)))

    for generation in range(max_generations):
        # Výpočet hodnôt funkcie pre každého jedinca
        fitness_values = np.array([objective_function(individual) for individual in population])

        # Aplikácia obmedzení
        for i, individual in enumerate(population):
            if not constraints(individual):
                fitness_values[i] = float('-inf')

        # Výber rodičov
        parents = population[np.argsort(fitness_values)[::-1][:2]]

        # Vytvorenie potomkov pomocou kríženia
        children = crossover(parents[0], parents[1], crossover_prob)

        # Aplikácia mutácie na potomkov
        children = mutate(children, bounds)

        # Nahradenie najhorších potomkov v populácii
        worst_indices = np.argsort(fitness_values)[:2]
        population[worst_indices] = children

        if draw_after_each_iteration:
            # Vykreslenie výsledkov po každej iterácii
            plot_results(bounds, population, Lx, Ly, r)

    # Nájdenie najlepšieho riešenia
    best_index = np.argmax(fitness_values)
    best_solution = population[best_index]

    if not draw_after_each_iteration:
        # Vykreslenie výsledkov na konci
        plot_results(bounds, population, Lx, Ly, r)

    return best_solution

def crossover(parent1, parent2, crossover_prob):
    # Uniformné kríženie
    mask = np.random.rand(len(parent1)) < crossover_prob
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return np.array([child1, child2])

def mutate(children, bounds, mutation_prob=0.1):
    # Mutácia v jednom bode
    mask = np.random.rand(*children.shape) < mutation_prob
    children += mask * np.random.uniform(low=-1, high=1, size=children.shape) * (bounds[:, 1] - bounds[:, 0])
    children = np.clip(children, bounds[:, 0], bounds[:, 1])
    return children

def objective_function(positions):
    # Funkcia, ktorú chceme maximalizovať (vzdialenosť dvoch najbližších stolov)
    total_distance = 0
    n = len(positions) // 2

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.sqrt((positions[i] - positions[j]) ** 2 + (positions[i + n] - positions[j + n]) ** 2)
            total_distance += max(0, distance - 2 * r)  # Berieme do úvahy polomer stola

    return -total_distance  # Minimizujeme negatívnu vzdialenosť, aby sme maximalizovali vzdialenosť

def constraints(positions):
    # Obmedzenia pre vzdialenosť stolu od hraníc obdĺžnika
    for i in range(len(positions) // 2):
        if positions[i] - r < 0 or positions[i] + r > Lx or positions[i + len(positions) // 2] - r < 0 or positions[i + len(positions) // 2] + r > Ly:
            return False
    return True

def plot_results(bounds, population, Lx, Ly, r):
    fig, ax = plt.subplots()
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)

    for pos in population:
        circle = plt.Circle((pos[0], pos[1]), r, fill=False, color='blue')
        ax.add_artist(circle)

    # Obdĺžnik predstavujúci kaviareň
    rectangle = plt.Rectangle((0, 0), bounds[0][1], bounds[1][1], fill=False, edgecolor='red', linewidth=2)
    ax.add_artist(rectangle)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Optimal Table Placement')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

if __name__ == "__main__":
    Lx = 20  # Šírka kaviarne
    Ly = 20  # Výška kaviarne
    r = 1    # Polomer stola
    n = 5    # Počet stolov

    bounds = np.array([[0, Lx], [0, Ly]] * n)
    optimal_solution = differential_evolution(objective_function, bounds, constraints, draw_after_each_iteration=True)

    print("Optimal positions:", optimal_solution)

