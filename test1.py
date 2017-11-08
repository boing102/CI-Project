from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import numpy as np
import pickle

data = np.genfromtxt('/home/student/Downloads/train_data/alpine-1.csv', skip_header=1, dtype=float, delimiter=',', skip_footer=1)
x_train = data[0:13600,3:25]
y_train = data[0:13600,0:3]
x_test = data[13600:,3:25]
y_test = data[13600:,0:3]

nn = MLPRegressor(
    hidden_layer_sizes=(28,28,28))

n = nn.fit(x_train, y_train)

# data_test = np.genfromtxt('/home/student/Downloads/train_data/aalborg.csv', skip_header=1, dtype=float, delimiter=',', skip_footer=1)
# x_test = data_test[0:500,0:3]
# y_test = data_test[0:500,3:25]
score = nn.score(x_test, y_test)
print('R2 score is (1.0 is best)', score)

with open('./models/sklearn.pickle', 'wb') as handle:
    pickle.dump(nn, handle)
