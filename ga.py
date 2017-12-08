import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from data import all_data, split_data, x_y

# For report:
# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

input_layer_nodes = 22
output_layer_nodes = 3
max_hidden_layers = 3  # Genome length.
min_hidden_layer_nodes = 0
max_hidden_layer_nodes = 5 * max(input_layer_nodes, output_layer_nodes)

# The probability of recombination being applied to a pair of parents.
prob_parents_recombining = 0.5
# The probability of a gene in a genome being mutated. p_m in literature.
prob_gene_mutation = 1 / max_hidden_layers  # 1 / l.
# The probability of a hidden layer being zero (empty) on mutation.
prob_zero_hidden_layer = 0.15
# Mu.
pop_size = 20
# Generational GA.
num_offspring = pop_size
# k.
tournament_size = 3
# Accuracy on training set.
termination_accuracy = 0.95
max_epochs = 2  # Default: 200.

all_data_ = all_data()

if pop_size % 2 != 0:
    raise AssertionError("Pop size should be a multiple of 2")


# Shuffle and set up training/testing data.
def shuffle_and_setup_data():
    np.random.shuffle(all_data_)
    train_data, _, test_data = split_data(all_data_, 4, 0, 1)
    global y_train
    global x_train_norm
    global y_test
    global x_test_norm
    x_train, y_train = x_y(train_data)
    x_train_norm = normalize(x_train)
    x_test, y_test = x_y(test_data)
    x_test_norm = normalize(x_test)
    pca = PCA(n_components=7)
    x_train_norm = pca.fit_transform(x_train_norm)
    x_test_norm = pca.transform(x_test_norm)


# A list of the nodes in hidden layers, and maybe a fitness.
class Genome:

    # A uniform random genome without fitness.
    def __init__(self):
        self.genome = [uniform_gene(high_zero_prob=True)
                       for _ in range(max_hidden_layers)]
        self.fitness = 0

    # Pretty information.
    def __str__(self):
        return "hidden layers: {0} fitness: {1}".format(
            self.hidden_layer_sizes(), self.fitness)

    # A list of hidden layer sizes.
    def hidden_layer_sizes(self):
        return list(filter(lambda x: x > 0, self.genome))

    # Evaluate the genome's fitness.
    def evaluate(self):
        shuffle_and_setup_data()
        nn = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes(), max_iter=max_epochs)
        nn.fit(x_train_norm, y_train)
        # Don't allow for negative fitness. 0 fitness individuals will not survive.
        self.fitness = max(0, nn.score(x_test_norm, y_test))


# A uniform random amount of nodes for a hidden layer.
# Zero (empty hidden layer) may be given a higher probability.
def uniform_gene(high_zero_prob=False) -> int:
    if high_zero_prob and np.random.rand() < prob_zero_hidden_layer:
        return 0
    return np.random.randint(
        min_hidden_layer_nodes, max_hidden_layer_nodes + 1)


# A randomly initialized population.
def rand_population(amount) -> [Genome]:
    return [Genome() for _ in range(amount)]


# Apply recombination to each pair of parents.
def recombinations_of(mating_pool):
    result = []
    for i in range(0, len(mating_pool) - 1, 2):
        result += recombine_maybe(mating_pool[i], mating_pool[i+1])
    assert len(mating_pool) == len(result)
    return result


# One point crossover, probabilistically applied, else returns parents.
def recombine_maybe(parent_a, parent_b) -> (Genome, Genome):
    # Recombination may not happen.
    if np.random.rand() >= prob_parents_recombining:
        return parent_a, parent_b
    # Okay it will happen.
    child_a = Genome()
    child_b = Genome()
    point = np.random.randint(1, max_hidden_layers)
    for i in range(max_hidden_layers):
        if i < point:
            child_a.genome[i], child_b.genome[i] = parent_a.genome[i], parent_b.genome[i]
        else:
            child_a.genome[i], child_b.genome[i] = parent_b.genome[i], parent_a.genome[i]
    return [child_a, child_b]


# Random resetting with independent probability p_m per gene.
def mutate_maybe(genome):
    for i in range(max_hidden_layers):
        if np.random.rand() < prob_gene_mutation:
            genome.genome[i] = uniform_gene(high_zero_prob=True)
    return genome


# Fitness proportional selection using SUS.
def parent_selection(population, amount) -> [Genome]:
    # Set proportional fitness on genomes.
    fitness_sum = sum([genome.fitness for genome in population])
    for genome in population:
        genome.p_sel = genome.fitness / fitness_sum
    # Set cumulative probability distribution.
    population = sorted(population, key=lambda x: x.fitness)
    p_sel_sum = 0
    cpd = [0 for _ in range(len(population))]
    for i, genome in enumerate(population):
        p_sel_sum += genome.p_sel
        cpd[i] = p_sel_sum

    # SUS.
    mating_pool = [None for _ in range(num_offspring)]
    current_member = 0
    i = 1
    r = np.random.uniform(0, 1 / num_offspring)
    while current_member < num_offspring:
        while current_member < num_offspring and r < cpd[i - 1]:
            mating_pool[current_member] = population[i - 1]
            current_member += 1
            r += 1 / num_offspring
        i += 1
    np.random.shuffle(mating_pool)

    return mating_pool[0:num_offspring]


# Tournament selection.
def survivor_selection(offspring):
    survivors = [None for _ in range(num_offspring)]
    for current_member in range(num_offspring):
        tournament = [np.random.choice(offspring) for _ in range(tournament_size)]
        survivors[current_member] = max(tournament, key=lambda x: x.fitness)
    return survivors


# Given a population, we decide to terminate if a fitness threshold is reached.
def will_terminate(population):
    fittest_genome = max(population, key=lambda g: g.fitness)
    print("Generation fittest genome = {0}".format(fittest_genome))
    return fittest_genome.fitness >= termination_accuracy


# Evaluate a population of individuals.
def evaluate(population):
    [genome.evaluate() for genome in population]


def assert_good_pop(population):
    for genome in population:
        assert genome is not None


def run():
    i = 0
    while i == 0 or not will_terminate(population):
        # Initialize
        if i == 0:
            print("Initialize random population")
            population = rand_population(pop_size)
            evaluate(population)
        print("Generation ", i)
        best = max(population, key=lambda x: x.fitness)
        mean = np.mean([x.fitness for x in population])
        # Store best individual of generation for future evaluation.
        with open("ga-results.txt", "a") as f:
            f.write("\n" + str(best))
        with open("ga-avg.txt", "a") as f:
            f.write("\n" + str(mean))
        print("population")
        [print(genome) for genome in population]
        assert_good_pop(population)

        # Mating pool
        mating_pool = parent_selection(population, num_offspring)
        print("mating_pool")
        [print(genome) for genome in mating_pool]

        # Offspring
        print("offspring")
        offspring = recombinations_of(mating_pool)
        evaluate(offspring)
        [print(genome) for genome in offspring]

        # Mutate offspring
        mutants = [mutate_maybe(genome) for genome in offspring]
        print("evaluate mutants")
        evaluate(mutants)
        [print(genome) for genome in mutants]

        # Survivor selection
        population = survivor_selection(mutants)
        evaluate(population)
        population[np.random.randint(len(population))] = best
        i += 1


if __name__ == "__main__":
    run()
