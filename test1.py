from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import sys
from sklearn.decomposition import PCA
from data import all_data, x_y, split_data


def train(folder):
    RANDOM_SEED = 5
    # RANDOM_SEED = np.random.randint(1)
    tr, _, te = split_data(all_data(folder=folder), 4, 0, 1)
    np.random.shuffle(tr)
    np.random.shuffle(te)
    x_train = tr[:, 3:]
    y_train = tr[:, :3]
    x_test = te[:, 3:]
    y_test = te[:, :3]
    pca = PCA(n_components=6)

    x_train_norm = normalize(x_train)
    x_test_norm = normalize(x_test)

    x_train_norm = pca.fit_transform(x_train_norm)
    x_test_norm = pca.transform(x_test_norm)

# # scaler = StandardScaler()
# # x_train_scale = scaler.fit_transform(x_train)
# # x_test_scale = scaler.transform(x_test)

# x_train_norm = normalize(x_train_scale)
# x_test_norm = normalize(x_test_scale)

    nn = MLPRegressor(
        hidden_layer_sizes=(27, 42, 43),
        random_state=RANDOM_SEED,
        learning_rate='invscaling')

    n = nn.fit(x_train_norm, y_train)

    score = nn.score(x_test_norm, y_test)
    print('R2 score is (1.0 is best)', score)

# prediction = nn.predict(x_test_norm[1:3,:])
# print("prediction ", prediction, " true ", y_test[1:3,:])

    # Previous behaviour.
    if folder == "train_data":
        with open('./models/sklearn.pickle', 'wb') as handle:
            pickle.dump(nn, handle)

        with open('./models/pca.pickle', 'wb') as handle:
            pickle.dump(pca, handle)

    else:
        with open('./models/' + folder + '_sklearn.pickle', 'wb') as handle:
            pickle.dump(nn, handle)

        with open('./models/' + folder + '_pca.pickle', 'wb') as handle:
            pickle.dump(pca, handle)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        train(sys.argv[1])
    else:
        train("train_data")