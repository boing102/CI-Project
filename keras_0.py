from keras import backend as K
from keras import Input, Model
from keras.layers import Dense, Activation, concatenate
from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import preprocessing as pp
import numpy as np
from data import all_data, x_y, split_data
import pickle

#Params
usePCA = False
standardize = True
input_dim = 22
n_epochs = 20

#Perform PCA
def PCAFunction(x):
    pcaVars = 7
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
    S = Input(shape = (input_dim,))
    h0 = Dense(units=22, activation = 'relu')(S)
    h1 = Dense(units=28,activation = 'relu')(h0)
    Steering = Dense(1)(h1)#activation='tanh')(h1)   
    Acceleration = Dense(1)(h1)#,activation='sigmoid')(h1)   
    Brake = Dense(1)(h1)#,activation='sigmoid')(h1)   
    V = concatenate([Steering,Acceleration,Brake], axis = 1)          
    model = Model(input=S,output=V)
    return model
'''
    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])  
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        V = merge([Steering,Acceleration,Brake],mode='concat')          
        model = Model(input=S,output=V)
        print("We finished building the model")
        return model, model.trainable_weights, S

'''

# A high level function that takes a model, trains & saves it.
def run_model(nn_model):
    tr, _, te = split_data(all_data(), 5, 1, 1)
    x_train, y_train = x_y(tr)
    x_test, y_test = x_y(te)
    xData,_ = x_y(all_data())
    scaler = pp.StandardScaler().fit(xData)
    if standardize:
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        print("Scalar means: ", x_train.mean(axis=0))
        print("Scalar sigma's: ", x_train.std(axis=0))
    if usePCA:
        x_train = PCAFunction(x_train)    
        x_test = PCAFunction(x_test)
    print("Xtrain is ", x_train.shape, type(x_train))
    print("Min/max of each column: ", x_test.min(axis=0), x_test.max(axis=0))
    model = nn_model(input_dim=x_train.shape[1])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=32)
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    print("Loss & accuracy on test data: {0}".format(loss_and_metrics))
    model.save("./models/keras.pickle")


if __name__ == "__main__":
    run_model(simple_NN)
    K.clear_session()
