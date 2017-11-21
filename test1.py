from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
import numpy as np
import pickle
from data import all_data, x_y, split_data

tr, _, te = split_data(all_data(), 4, 0, 1)
x_train = tr[:,3:25]
y_train = tr[:,0:3]
x_test = te[:,3:25]
y_test = te[:,0:3]

x_train_norm = normalize(x_train)
x_test_norm = normalize(x_test)
print(x_train_norm.shape)
nn = MLPRegressor(
    hidden_layer_sizes=(50,50, 50, 50), max_iter=1000)

n = nn.fit(x_train_norm, y_train)

# data_test = np.genfromtxt('/home/student/Downloads/train_data/aalborg.csv', skip_header=1, dtype=float, delimiter=',', skip_footer=1)
# x_test = data_test[0:500,0:3]
# y_test = data_test[0:500,3:25]
score = nn.score(x_test_norm, y_test)
print('R2 score is (1.0 is best)', score)

# prediction = nn.predict(x_test_norm[1:3,:])
# print("prediction ", prediction, " true ", y_test[1:3,:])

with open('./models/sklearn.pickle', 'wb') as handle:
    pickle.dump(nn, handle)
