from keras.models import Sequential
from keras.layers import Dense, Activation

from data import all_data, x_y, split_data


# A simple NN model.
def simple_NN(input_dim):
    model = Sequential()
    model.add(Dense(units=22, input_shape=(input_dim,)))
    model.add(Activation("relu"))
    model.add(Dense(units=22))
    model.add(Activation("relu"))
    model.add(Dense(units=3))
    model.add(Activation("softmax"))
    return model


# A high level function that takes a model and evaluates it.
def run_model(model):
    tr, va, te = split_data(all_data(), 5, 1, 1)
    x_train, y_train = x_y(tr)
    print("x-train: {0}".format(x_train.shape))
    print("y-train: {0}".format(y_train.shape))
    x_test, y_test = x_y(te)
    print("x-test: {0}".format(x_test.shape))
    print("y-test: {0}".format(y_test.shape))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)


if __name__ == "__main__":
    model = simple_NN(22)
    run_model(model)
