from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
import numpy as np
import pickle
from data import all_data, x_y, split_data

# data = np.genfromtxt('./train_data/alpine-1.csv', skip_header=1, dtype=float, delimiter=',', skip_footer=1)
# x_train = data[0:13600,3:25]
# y_train = data[0:13600,0:3]
# x_test = data[13600:,3:25]
# y_test = data[13600:,0:3]
tr, _, te = split_data(all_data(), 4, 0, 1)

# tr_acce = tr[ (0<tr[:,0])]
# tr_braking = tr[ (0<tr[:,1])]
# np.random.shuffle(tr_acce)
# tr_acce = tr_acce[0:len(tr_braking),:]
# tr = np.concatenate((tr_acce, tr_braking))
# np.random.shuffle(tr)

x_train = tr[:,3:25]
y_train = tr[:,0:3]
x_test = te[:,3:25]
y_test = te[:,0:3]

x_train_norm = normalize(x_train)
x_test_norm = normalize(x_test)

regr_rf = RandomForestRegressor(max_depth=20, random_state=2)
regr_rf.fit(x_train_norm, y_train)

# data_test = np.genfromtxt('/home/student/Downloads/train_data/aalborg.csv', skip_header=1, dtype=float, delimiter=',', skip_footer=1)
# x_test = data_test[0:500,0:3]
# y_test = data_test[0:500,3:25]
score = regr_rf.score(x_test_norm, y_test)
print('R2 score is (1.0 is best)', score)

prediction = regr_rf.predict(x_test_norm[1:3,:])
print("prediction ", prediction, " true ", y_test[1:3,:])

with open('./models/rf.pickle', 'wb') as handle:
    pickle.dump(regr_rf, handle)
