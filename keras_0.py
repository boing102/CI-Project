from keras import backend as K
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np
from data import all_data, x_y, split_data

#input_dim = 22
pcaVars = 5


def PCAFunction(x):
	x = scale(x)
	pca = PCA(n_components = 22)
	pca.fit(x)
	var1 = np.cumsum(np.round(pca.explained_variance_ratio_,decimals =4)*100)
	'''	
	#Check how many variables we want to use
	for i in range(len(var1)):
		if var1[i] > threshold:
			break
	print(i)	
	'''
	pcaRed = PCA(n_components = pcaVars)
	return pcaRed.fit_transform(x)	


# A simple NN model.
def simple_NN():
    model = Sequential()
    model.add(Dense(units=50, input_shape=(pcaVars,)))
    model.add(Activation("relu"))
    model.add(Dense(units=25))
    model.add(Activation("relu"))
    model.add(Dense(units=3))
    model.add(Activation("softmax"))
    return model


# A high level function that takes a model, trains & saves it.
def run_model(model):
    tr, _, te = split_data(all_data(), 5, 1, 1)
    x_train, y_train = x_y(tr)
    print(x_train[:3])
    x_train = PCAFunction(x_train)
    print(x_train[:3])

    print("x-train: {0}".format(x_train.shape))
    print("y-train: {0}".format(y_train.shape))
    x_test, y_test = x_y(te)
    x_test = PCAFunction(x_test)
    print("x-test: {0}".format(x_test.shape))
    print("y-test: {0}".format(y_test.shape))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    print(loss_and_metrics)
    model.save("./models/keras.pickle")


if __name__ == "__main__":
    model = simple_NN()
    run_model(model)
    K.clear_session()
