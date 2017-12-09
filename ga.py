import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from data import all_data, split_data, x_y

# For report:
# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

input_layer_nodes = 22
output_layer_nodes = 3
min_hidden_layers = 1
max_hidden_layers = 3  # Genome length.
min_hidden_layer_nodes = 0
max_hidden_layer_nodes = 5 * max(input_layer_nodes, output_layer_nodes)

# The probability of recombination being applied to a pair of parents.
prob_parents_recombining = 0.5
# The probability of a gene in a genome being mutated. p_m in literature.
# 1 / l is a probability of 1 genome on average.
prob_gene_mutation = 1 / max_hidden_layers
# The probability of a hidden layer being zero (empty) on mutation.
prob_zero_hidden_layer = 0.15
use_high_prob_zero_hl = False
# Probability of injecting a new random individual. Essentially this is a
# second mutation operator that applies mutation with a probability to the
# entire genome, with a uniform random amount of hidden layers.
mutate_entire_genome_prob = 0.1  # So 5 / 50 genomes.
use_genome_mutation = True
# Mu.
pop_size = 50
# Generational GA.
num_offspring = pop_size
# k.
tournament_size = 10
max_epochs = 1  # Default: 200.
SHUFFLE = True  # Shuffle training data.
REPEAT = 5  # Average fitness over REPEAT times.

all_data_ = all_data()
train_data, _, test_data = split_data(all_data_, 4, 0, 1)

if pop_size % 2 != 0:
    raise AssertionError("Pop size should be a multiple of 2")


# Shuffle and set up training/testing data.
def shuffle_and_setup_data(shuffle):
    if shuffle:
        np.random.shuffle(train_data)
    # Should always shuffle test data? Just in-case.
    np.random.shuffle(test_data)
    global y_train
    global x_train_norm
    global y_test
    global x_test_norm
    x_train, y_train = x_y(train_data)
    x_train_norm = normalize(x_train)
    x_test, y_test = x_y(test_data)
    x_test_norm = normalize(x_test)
    pca = PCA(n_components=6)
    x_train_norm = pca.fit_transform(x_train_norm)
    x_test_norm = pca.transform(x_test_norm)


# A list of the nodes in hidden layers, and maybe a fitness.
class Genome:

    # A uniform random genome without fitness.
    def __init__(self):
        num_hidden_layers = np.random.randint(
            min_hidden_layers, max_hidden_layers + 1)
        print("random hidden layers: {0}".format(num_hidden_layers))
        self.genome = [uniform_gene() for _ in range(num_hidden_layers)]
        # Pad missing values with 0.
        self.genome += [0 for _ in range(max_hidden_layers - len(self.genome))]
        assert len(self.genome) == max_hidden_layers
        self.fitness = 0

    # Pretty information.
    def __str__(self):
        return "hidden layers: {0} fitness: {1}".format(
            self.hidden_layer_sizes(), self.fitness)

    # A list of hidden layer sizes.
    def hidden_layer_sizes(self):
        return list(filter(lambda x: x > 0, self.genome))

    # Evaluate the genome's fitness.
    def evaluate(self, shuffle=SHUFFLE, max_iter=max_epochs, repeat=REPEAT):
        self.fitness = np.mean([self.single_eval(shuffle, max_iter)
                                for _ in range(repeat)])

    def single_eval(self, shuffle, max_iter):
        shuffle_and_setup_data(shuffle=shuffle)
        nn = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes(), max_iter=max_iter)
        nn.fit(x_train_norm, y_train)
        # Don't allow for negative fitness. 0 fitness individuals will not survive.
        return max(0, nn.score(x_test_norm, y_test))


# A uniform random amount of nodes for a hidden layer.
# Zero (empty hidden layer) may be given a higher probability.
def uniform_gene() -> int:
    if use_high_prob_zero_hl and np.random.rand() < prob_zero_hidden_layer:
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
    if np.random.rand() > prob_parents_recombining:
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
    # We may mutate the entire genome, in which case we ignore the other
    # mutation operator.
    if use_genome_mutation and np.random.rand() < mutate_entire_genome_prob:
        print("Genome prior entire mutation: {0}".format(genome))
        genome = Genome()
        print("Genome after entire mutation: {0}".format(genome))
    # Else we apply gaussian perturbation to each gene independently with
    # probability prob_zero_hidden_layer. 0 has a high probability.
    for i in range(max_hidden_layers):
        if np.random.rand() < prob_gene_mutation:
            genome.genome[i] = gene_from_gaussian_peturbation(genome.genome[i])
        if use_high_prob_zero_hl and np.random.rand() < prob_zero_hidden_layer:
            genome.genome[i] = 0
    return genome


# A gene from mutation using gaussian peturbation.
# Receives the gene's previous value.
def gene_from_gaussian_peturbation(val):
    print("val before peturbation: {0}".format(val))
    # Mean 0, spread is half of max range.
    gauss = int(np.random.normal(0, max_hidden_layer_nodes / 2))
    print("gauss: {0}".format(gauss))
    val += gauss
    print("val: {0}".format(val))
    val = max(0, min(max_hidden_layer_nodes, val))
    print("val after peturbation: {0}".format(val))
    return int(val)


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


# Evaluate a population of individuals.
def evaluate(population):
    [genome.evaluate() for genome in population]


# Just a sanity-check that genomes are not None.
def assert_good_pop(population):
    for genome in population:
        assert genome is not None


# Main loop of this neuro-topology optimizing GA.
def run():
    i = 0
    while True:
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


# Functions for testing different settings of the EA ##########################


# How often to repeat given max_iter=1.
def test_repeat():
    for repeat in range(1, 20):
        diff = eval_diff(shuffle=False, max_iter=1, repeat=repeat)
        print("repeat {1} diff {0}".format(diff, repeat))


# Determine eval_diff over different max_iter.
def test_max_iter():
    for max_iter in range(1, 20):
        diff = eval_diff(shuffle=False, max_iter=max_iter, repeat=REPEAT)
        print("max_iter {0}: diff {1}".format(max_iter, diff))


# Should use shuffle or not?
def test_shuffle():
    total_shuffle_diff = 0
    total_no_shuffle_diff = 0
    for max_iter in range(5, 10):
        shuffle_diff = eval_diff(shuffle=True, max_iter=max_iter, repeat=5)
        no_shuffle_diff = eval_diff(shuffle=False, max_iter=max_iter, repeat=5)
        print("shuffle {0} max_iter {1} diff {2}".format(True, max_iter, shuffle_diff))
        print("shuffle {0} max_iter {1} diff {2}".format(False, max_iter, no_shuffle_diff))
        total_shuffle_diff += shuffle_diff
        total_no_shuffle_diff += no_shuffle_diff
    print("total shuffle diff {0}".format(total_shuffle_diff))
    print("total no shuffle diff {0}".format(total_no_shuffle_diff))


# Difference in two equivalent evaluations of a random genome.
def eval_diff(shuffle, max_iter, repeat):
    genome = Genome()
    genome.evaluate(shuffle=shuffle, max_iter=max_iter, repeat=repeat)
    f1 = genome.fitness
    genome.evaluate(shuffle=shuffle, max_iter=max_iter, repeat=repeat)
    f2 = genome.fitness
    return abs(f1 - f2)

###############################################################################


if __name__ == "__main__":
    # test_shuffle()
    # test_repeat()
    # test_max_iter()
    run()
