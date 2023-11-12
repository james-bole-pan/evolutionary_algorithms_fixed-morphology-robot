import numpy as np
import random
import matplotlib.pyplot as plt

# Constants
g = np.array([0.0, 0.0, -9.81])  # Gravity
dt = 0.0001
k = 1000.0  # Spring constant
L0 = 1.0  # Rest length of the spring
damping = 0.75  # Damping constant
mu_s = 1.0  # Static friction coefficient
mu_k = 0.8  # Kinetic friction coefficient
half_L0 = L0/2
drop_height = 1.0
omega = 2*np.pi*2 # frequency of breathing
times_of_simulation = 10000
mutation_range_k = [1000, 1200]
mutation_range_b = [0.2, 0.3]
mutation_range_c = [0, 2*np.pi*0.1]
mutation_probability = 0.2
crossover_probability = 0.5
population_size = 10
generations = 15

class Mass:
    def __init__(self, p, v, m=0.1):
        self.m = m
        self.p = np.array(p)
        self.v = np.array(v)
        self.a = np.zeros(3,dtype=float)
        self.f = np.zeros(3,dtype=float)

class Spring:
    def __init__(self, L0, k, m1, m2):
        self.L0 = L0
        self.k = k
        self.m1 = m1
        self.m2 = m2

class Individual:
    def __init__(self):
        masses = [
            Mass([-half_L0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # 0
            Mass([half_L0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 1
            Mass([-half_L0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 2
            Mass([half_L0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 3
            Mass([-half_L0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 4
            Mass([half_L0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 5
            Mass([-half_L0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 6
            Mass([half_L0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),      # 7
            Mass([-half_L0, half_L0 + L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # 8
            Mass([half_L0, half_L0 + L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 9
            Mass([-half_L0, half_L0 + L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 10
            Mass([half_L0, half_L0 + L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 11
        ]
        short_diag_length = np.sqrt(2 * L0**2)
        long_diag_length = np.sqrt(3 * L0**2)

        springs = [
            Spring(L0, k, masses[0], masses[1]),  # Base square
            Spring(L0, k, masses[1], masses[3]),
            Spring(L0, k, masses[3], masses[2]),
            Spring(L0, k, masses[2], masses[0]),
            Spring(L0, k, masses[4], masses[5]),  # Top square
            Spring(L0, k, masses[5], masses[7]),
            Spring(L0, k, masses[7], masses[6]),
            Spring(L0, k, masses[6], masses[4]),
            Spring(L0, k, masses[0], masses[4]),  # Vertical edges
            Spring(L0, k, masses[1], masses[5]),
            Spring(L0, k, masses[2], masses[6]),
            Spring(L0, k, masses[3], masses[7]),
            Spring(short_diag_length, k, masses[0], masses[3]),
            Spring(short_diag_length, k, masses[1], masses[2]),
            Spring(short_diag_length, k, masses[4], masses[7]),
            Spring(short_diag_length, k, masses[5], masses[6]),
            Spring(short_diag_length, k, masses[0], masses[5]),
            Spring(short_diag_length, k, masses[1], masses[4]),
            Spring(short_diag_length, k, masses[2], masses[7]),
            Spring(short_diag_length, k, masses[3], masses[6]),
            Spring(short_diag_length, k, masses[1], masses[7]),
            Spring(short_diag_length, k, masses[0], masses[6]),
            Spring(short_diag_length, k, masses[3], masses[5]),
            Spring(short_diag_length, k, masses[2], masses[4]),
            Spring(long_diag_length, k, masses[0], masses[7]),
            Spring(long_diag_length, k, masses[1], masses[6]),
            Spring(long_diag_length, k, masses[2], masses[5]),
            Spring(long_diag_length, k, masses[3], masses[4]),
            Spring(L0, k, masses[8], masses[9]), 
            Spring(L0, k, masses[9], masses[11]),
            Spring(L0, k, masses[11], masses[10]),
            Spring(L0, k, masses[10], masses[8]),
            Spring(L0, k, masses[6], masses[10]),
            Spring(L0, k, masses[7], masses[11]),
            Spring(L0, k, masses[4], masses[8]),
            Spring(L0, k, masses[5], masses[9]),
            Spring(short_diag_length, k, masses[6], masses[11]),
            Spring(short_diag_length, k, masses[7], masses[10]),
            Spring(short_diag_length, k, masses[4], masses[9]),
            Spring(short_diag_length, k, masses[5], masses[8]),
            Spring(short_diag_length, k, masses[4], masses[10]),
            Spring(short_diag_length, k, masses[5], masses[11]),
            Spring(short_diag_length, k, masses[6], masses[8]),
            Spring(short_diag_length, k, masses[7], masses[9]),
            Spring(short_diag_length, k, masses[9], masses[10]),
            Spring(short_diag_length, k, masses[8], masses[11]),
            Spring(long_diag_length, k, masses[6], masses[9]),
            Spring(long_diag_length, k, masses[7], masses[8]),
            Spring(long_diag_length, k, masses[4], masses[11]),
            Spring(long_diag_length, k, masses[5], masses[10])
        ]
        self.masses = masses
        self.springs = springs
        self.a_dict = {}
        for spring in springs:
            self.a_dict[spring] = spring.L0
        self.b_dict = {spring:0.0 for spring in springs}
        self.c_dict = {spring:0.0 for spring in springs}
        self.k_dict = {spring:k for spring in springs}

    def set_b_dict(self, b_dict):
        self.b_dict = b_dict
    
    def set_c_dict(self, c_dict):
        self.c_dict = c_dict

    def set_k_dict(self, k_dict):
        self.k_dict = k_dict

def fitness(individual):
    masses = individual.masses
    springs = individual.springs
    initial_center_of_mass = p_center_of_mass(masses)
    for _ in range(times_of_simulation):
        simulation_step(masses, springs, dt, individual.a_dict, individual.b_dict, individual.c_dict, individual.k_dict)
    final_center_of_mass = p_center_of_mass(masses)
    displacement = final_center_of_mass - initial_center_of_mass
    speed = np.linalg.norm(displacement[:2]) / (times_of_simulation * dt) # only care about horizontal distance
    return speed

def p_center_of_mass(masses):
    return sum([mass.m * mass.p for mass in masses]) / sum([mass.m for mass in masses])

def get_cube_faces(masses):
    return [
        [masses[0].p, masses[1].p, masses[5].p, masses[4].p],  # Bottom face
        [masses[2].p, masses[3].p, masses[7].p, masses[6].p],  # Top face
        [masses[0].p, masses[1].p, masses[3].p, masses[2].p],  # Front face
        [masses[0].p, masses[4].p, masses[6].p, masses[2].p],  # Left face
        [masses[1].p, masses[5].p, masses[7].p, masses[3].p],   # Right face
        [masses[6].p, masses[7].p, masses[11].p, masses[10].p],
        [masses[7].p, masses[11].p, masses[9].p, masses[5].p],
        [masses[5].p, masses[9].p, masses[8].p, masses[4].p],
        [masses[4].p, masses[8].p, masses[10].p, masses[6].p],
        [masses[9].p, masses[11].p, masses[10].p, masses[8].p],
        [masses[6].p, masses[7].p, masses[5].p, masses[4].p]
    ]

def get_floor_tile():
    floor_size = 2.5
    return [[-floor_size, -floor_size, 0], 
            [floor_size, -floor_size, 0], 
            [floor_size, floor_size, 0], 
            [-floor_size, floor_size, 0]]

def simulation_step(masses, springs, dt, a_dict, b_dict, c_dict, k_dict):
    global t
    t += dt

    # Reset forces on each mass
    for mass in masses:
        mass.f = np.zeros(3, dtype=float)
        mass.f += mass.m * g  # Gravity

    # Calculate spring forces
    for spring in springs:
        a = a_dict[spring]
        b = b_dict[spring]
        c = c_dict[spring]
        spring.k = k_dict[spring]
        spring.L0 = a + b*np.sin(omega*t+c) 

        delta_p = spring.m1.p - spring.m2.p
        delta_length = np.linalg.norm(delta_p)
        if delta_length == 0:
            direction = np.zeros(3, dtype=float)
        else:
            direction = delta_p / delta_length
        force_magnitude = spring.k * (delta_length - spring.L0)
        force = force_magnitude * direction

        # Apply spring force to masses
        spring.m1.f -= force
        spring.m2.f += force

    # tally friction
    for mass in masses:
        if mass.p[2] > 0:
            continue
        F_n = mass.m * g[2]
        F_H = np.linalg.norm(mass.f[:2])
        direction = mass.f[:2] / F_H
        if F_n < 0:
            if F_H<=-mu_s*F_n:
                mass.f[:2] = np.zeros(2)
                print("static friction, ", mass.f)
            else:
                mass.f[:2] += -abs(mu_k*F_n)*direction
                print("kinetic friction, ", mass.f)

    # Update positions and velocities for each mass
    for mass in masses:
        mass.a = mass.f / mass.m
        mass.v += mass.a * dt
        mass.p += mass.v * dt

        # Simple collision with the ground
        if mass.p[2] < 0:
            mass.p[2] = 0
            mass.v[2] = -damping * mass.v[2]  # Some damping on collision

def mutation(individual):
    b_dict = individual.b_dict
    c_dict = individual.c_dict
    k_dict = individual.k_dict
    for spring in individual.springs:
        if random.random() < mutation_probability:
            b_dict[spring] = np.random.uniform(mutation_range_b[0], mutation_range_b[1])
            c_dict[spring] = np.random.uniform(mutation_range_c[0], mutation_range_c[1])
            k_dict[spring] = np.random.uniform(mutation_range_k[0], mutation_range_k[1])
    individual.set_b_dict(b_dict)
    individual.set_c_dict(c_dict)
    individual.set_k_dict(k_dict)
    return individual

def crossover(individual1, individual2, crossover_probability=0.5):
    child1 = Individual()
    child2 = Individual()

    # It's assumed that both individuals have springs in the same order
    # and represent the same physical connections.
    num_springs = len(individual1.springs)
    
    for i in range(num_springs):
        if random.random() < crossover_probability:
            # Child 1 gets parameter from individual 2 and vice versa for child 2
            child1.b_dict[child1.springs[i]] = individual2.b_dict[individual2.springs[i]]
            child1.c_dict[child1.springs[i]] = individual2.c_dict[individual2.springs[i]]
            child1.k_dict[child1.springs[i]] = individual2.k_dict[individual2.springs[i]]
            
            child2.b_dict[child2.springs[i]] = individual1.b_dict[individual1.springs[i]]
            child2.c_dict[child2.springs[i]] = individual1.c_dict[individual1.springs[i]]
            child2.k_dict[child2.springs[i]] = individual1.k_dict[individual1.springs[i]]
        else:
            # Child 1 gets parameter from individual 1 and vice versa for child 2
            child1.b_dict[child1.springs[i]] = individual1.b_dict[individual1.springs[i]]
            child1.c_dict[child1.springs[i]] = individual1.c_dict[individual1.springs[i]]
            child1.k_dict[child1.springs[i]] = individual1.k_dict[individual1.springs[i]]
            
            child2.b_dict[child2.springs[i]] = individual2.b_dict[individual2.springs[i]]
            child2.c_dict[child2.springs[i]] = individual2.c_dict[individual2.springs[i]]
            child2.k_dict[child2.springs[i]] = individual2.k_dict[individual2.springs[i]]

    return child1, child2
'''
# create a population
population = [Individual() for _ in range(population_size)]

for individual in population:
    b_dict = {}
    c_dict = {}
    k_dict = {}

    for spring in individual.springs:
        b_dict[spring] = np.random.uniform(mutation_range_b[0], mutation_range_b[1])
        c_dict[spring] = np.random.uniform(mutation_range_c[0], mutation_range_c[1])
        k_dict[spring] = np.random.uniform(mutation_range_k[0], mutation_range_k[1])

    individual.set_b_dict(b_dict)
    individual.set_c_dict(c_dict)
    individual.set_k_dict(k_dict)

t = 0

fitness_history = []
fitnesses = [fitness(individual) for individual in population]
max_fitness = max(fitnesses)
fitness_history.append(max_fitness)
best_individual = population[fitnesses.index(max_fitness)]

for i in range(generations):
    # select parents
    children = []
    while len(children) < population_size:
        parent1 = random.choices(population, weights=fitnesses)[0]
        parent2 = random.choices(population, weights=fitnesses)[0]
    
        child1, child2 = crossover(parent1, parent2)
        children.append(mutation(child1))
        children.append(mutation(child2))

    # evaluate fitness
    fitnesses = [fitness(individual) for individual in population]
    current_max_fitness = max(fitnesses)
    current_best_individual = population[fitnesses.index(current_max_fitness)]
    if current_max_fitness > max_fitness:
        max_fitness = current_max_fitness
        best_individual = current_best_individual
    fitness_history.append(max_fitness)
    print("Generation ", i, " fitnesses: ", fitnesses)
    with open("dot_plot_data.txt", "a") as f:
        f.write(str(fitnesses) + "\n")

print("Best individual: ", best_individual)
print("Best fitness: ", max_fitness)

best_b_dict = best_individual.b_dict
best_c_dict = best_individual.c_dict
best_k_dict = best_individual.k_dict
# save the best individual's parameters
np.save("best_b_dict.npy", best_b_dict)
np.save("best_c_dict.npy", best_c_dict)
np.save("best_k_dict.npy", best_k_dict)
'''
# implement a random search
population = [Individual() for _ in range(population_size)]

for individual in population:
    b_dict = {}
    c_dict = {}
    k_dict = {}

    for spring in individual.springs:
        b_dict[spring] = np.random.uniform(mutation_range_b[0], mutation_range_b[1])
        c_dict[spring] = np.random.uniform(mutation_range_c[0], mutation_range_c[1])
        k_dict[spring] = np.random.uniform(mutation_range_k[0], mutation_range_k[1])

    individual.set_b_dict(b_dict)
    individual.set_c_dict(c_dict)
    individual.set_k_dict(k_dict)

t = 0

rs_fitness_history = []
fitnesses = [fitness(individual) for individual in population]
max_fitness = max(fitnesses)
rs_fitness_history.append(max_fitness)
best_individual = population[fitnesses.index(max_fitness)]

for i in range(generations):
    new_population = [Individual() for _ in range(population_size)]
    for individual in new_population:
        b_dict = {}
        c_dict = {}
        k_dict = {}

        for spring in individual.springs:
            b_dict[spring] = np.random.uniform(mutation_range_b[0], mutation_range_b[1])
            c_dict[spring] = np.random.uniform(mutation_range_c[0], mutation_range_c[1])
            k_dict[spring] = np.random.uniform(mutation_range_k[0], mutation_range_k[1])

        individual.set_b_dict(b_dict)
        individual.set_c_dict(c_dict)
        individual.set_k_dict(k_dict)
    fitnesses = [fitness(individual) for individual in new_population]
    current_max_fitness = max(fitnesses)
    print("RS generation ", i, "current max fitness: ", current_max_fitness)
    current_best_individual = population[fitnesses.index(current_max_fitness)]
    if current_max_fitness > max_fitness:
        max_fitness = current_max_fitness
        best_individual = current_best_individual
    rs_fitness_history.append(max_fitness)

#plt.plot(fitness_history, label="EA")
plt.plot(rs_fitness_history, label="RS")
plt.xlabel("Generation")
plt.ylabel("Fitness (Speed of robot in m/s)")
plt.title("Fitness over generations")
plt.legend()
plt.show()