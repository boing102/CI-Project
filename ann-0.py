# import mnist_loader as mnist
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import sys


# Our activation function "f".
def activation(z):
    return 1 / (1 + np.exp(-z))


# The first derivative of the activation function.
def activation_f1(z):
    return activation(z) * (1 - activation(z))


# Our initial neural network class. A rewrite of code in the first two chapters
# of Michael Heath's book: http://neuralnetworksanddeeplearning.com (1)
class NN(object):

    # An NN with randomly initialized weights and biases.
    # Args:
    #   sizes: [int], a list containing the size of each hidden layer.
    def __init__(self, sizes, performance=None):
        self.performance = performance
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.layers = len(sizes)
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]
        self.weights = [np.random.randn(size_k, size_l)
                        for size_l, size_k in zip(sizes[:-1], sizes[1:])]

    # Apply the feed forward algorithm to current (W, b).
    def feed_forward(self, a):
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
        return a

    # Apply stochastic gradient descent for given amount of epochs. The
    # algorithm is stochastic because we only compute the cost derivative over
    # a subset of the data.
    def stochastic_gradient_descent(self, training_data, epochs, batch_size,
                                    learning_rate, test_data=None):
        # Until we have completed epoch epochs. One epoch is a pass over the
        # entire training data.
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            # We split the training_data into "len(training_data) / batch_size"
            # random batches, which is equal to the amount of gradient descent
            # steps taken. One step is a single adjustment of (W, b) based .
            batches = [training_data[x:x + batch_size]
                       for x in range(0, len(training_data), batch_size)]
            for batch in batches:
                self.gradient_descent_step(batch, learning_rate)
            print("Epoch {0} complete" + str(epoch))
            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(epoch))

    # Apply a gradient descent step for the given training data.
    def gradient_descent_step(self, batch, learning_rate):
        # We compute the sums of the rate of change of cost with respect to
        # each bias, and with respect to each weight. The sums are calculated
        # over the batch of training data.
        dC_dw_sums = [np.zeros(w.shape) for w in self.weights]
        dC_db_sums = [np.zeros(b.shape) for b in self.biases]
        # One element of training data, one forward and one backward pass.
        for x, y in batch:
            dC_db, dC_dw = self.back_propagation(x, y)
            dC_dw_sums += dC_dw
            dC_db_sums += dC_db
            # delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            dC_dw_sums = [nw+dnw for nw, dnw in zip(dC_dw_sums, dC_dw)]
            dC_db_sums = [nb+dnb for nb, dnb in zip(dC_db_sums, dC_db)]
        # We then compute the average of the rates of change, and multiply the
        # existing biases and weights by "-learning_rate * cost_delta_average".
        self.biases = [b - (learning_rate / len(batch)) * dC_db_sum
                       for b, dC_db_sum in zip(self.biases, dC_db_sums)]
        self.weights = [w - (learning_rate / len(batch)) * dC_dw_sum
                        for w, dC_dw_sum in zip(self.weights, dC_dw_sums)]

    # Returns the rate of change of the cost with respect to each bias and
    # weight.
    def back_propagation(self, x, y):

        # Forward pass #

        a = x  # Initial a is just the input.
        activations = [x]  # All the activations, layer by layer.
        zs = []  # All the z vectors, layer by layer.

        # For each layer calculate z and a=activation(z).
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = activation(z)
            activations.append(a)

        # Backward pass #

        dC_db = [np.zeros(b.shape) for b in self.biases]
        dC_dw = [np.zeros(w.shape) for w in self.weights]

        dC_dz = self.cost_f1(activations[-1], y) * activation_f1(zs[-1])
        dC_db[-1] = dC_dz
        dC_dw[-1] = np.dot(dC_dz, activations[-2].transpose())

        # Starting with the second last layer and going backwards.
        for l in range(2, self.num_layers):
            z = zs[-l]
            a = activation_f1(z)
            dC_dz = np.dot(self.weights[-l+1].transpose(), dC_dz) * a
            dC_db[-l] = dC_dz
            dC_dw[-l] = np.dot(dC_dz, activations[-l-1].transpose())
        return (dC_db, dC_dw)

    # Return the count of test inputs for which the NN guesses correctly.
    def evaluate(self, test_data):
        # test_results = [(np.argmax(self.feed_forward(x)), y)
        #                 for x, y in test_data]
        # return sum(int(x == y) for x, y in test_results)
        test_results = [(self.feed_forward(x), y) for x, y in test_data]
        return self.performance(test_results)
        return sum(metrics.mean_squared_error(x, y) for x, y in test_results)

    # Rate of change of cost with respect to activation.
    def cost_f1(self, output_activations, y):
        return 2 * (output_activations - y)


# Train a Neural network for mnist data.
def train_mnist():
    perf = lambda results: sum(int(np.argmax(x) == y) for x, y in results)
    net = NN([784, 30, 10], performance=perf)
    # training_data, validation_data, test_data = mnist.load_data_wrapper()
    training_data, validation_data, test_data = (None, None, None)
    print("ERROR: This is for MNIST not TORCS.")
    training_data = list(training_data)  # Generator to list.
    validation_data = list(validation_data)  # Generator to list.
    test_data = list(test_data)  # Generator to list.
    net.stochastic_gradient_descent(training_data, 30, 10, 3.0,
                                    test_data=test_data)


def load_torcs(filename):
    csv = pd.read_csv(filename)
    data = []
    for _, row in csv.iterrows():
        data.append((row[3:].reshape(21, 1), row[:3].reshape(3, 1)))
    return data


# Train a neural network for TORCS.
def train_torcs():
    perf = lambda results: sum(sum(np.isclose(x, y, rtol=0.1) for x, y in results))
    net = NN([21, 21, 21, 3], performance=perf)
    training_data = load_torcs("train_data/aalborg.csv")[:4360]  # MUST BE A MULTIPLE OF batch_size.
    net.stochastic_gradient_descent(training_data, 30, 10, 3.0,
                                    test_data=training_data)


if __name__ == "__main__":
    if "--mnist" in sys.argv:
        train_mnist()
    else:
        train_torcs()
