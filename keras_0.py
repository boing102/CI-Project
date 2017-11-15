from keras import backend as K
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import preprocessing as pp
import numpy as np
from data import all_data, x_y, split_data

#Params
pcaVars = 5
input_dim = pcaVars


def PCAFunction(x):
    pcaVars = 5
    x = scale(x)
    pca = PCA(n_components=22)
    pca.fit(x)
    var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    print(var1)
    '''
    #Check how many variables we want to use
    for i in range(len(var1)):
        if var1[i] > threshold:
            break
    print(i)
    '''
    pcaRed = PCA(n_components=pcaVars)
    return pcaRed.fit_transform(x)


# A simple NN model.
def simple_NN(input_dim):
    print("Neural network has {0} inputs".format(input_dim))
    model = Sequential()
    model.add(Dense(units=50, input_shape=(input_dim,)))
    model.add(Activation("relu"))
    model.add(Dense(units=25))
    model.add(Activation("relu"))
    model.add(Dense(units=3))
    model.add(Activation("softmax"))
    return model

# A high level function that takes a model, trains & saves it.
def run_model(nn_model):
    tr, _, te = split_data(all_data(), 5, 1, 1)
    x_train_temp, y_train = x_y(tr)
    xData,_ = x_y(all_data())
    #Normalise data
    scaler = pp.StandardScaler().fit(xData)
    x_train = scaler.transform(x_train_temp)
    print("Scalar means: ", x_train.mean(axis=0))
    print("Scalar sigma's: ", x_train.std(axis=0))
    # x_train = x_train[0].reshape(1, 22)
    # print("x-train reshaped: {0}".format(x_train.shape))
    x_train = PCAFunction(x_train)
    print("Xtrain is ", x_train.shape, type(x_train))
    x_test_temp, y_test = x_y(te)
    x_test = scaler.transform(x_test_temp)
    x_test = PCAFunction(x_test)
    model = nn_model(input_dim=x_train.shape[1])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=40, batch_size=32)
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    print("Loss & accuracy on test data: {0}".format(loss_and_metrics))
    model.save("./models/keras.pickle")


if __name__ == "__main__":
    run_model(simple_NN)
    K.clear_session()
