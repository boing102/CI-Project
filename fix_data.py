import glob
import os
import numpy as np


# Simplify a row of data.
def simplify_row(row):
    not_opponents, opponents = row[:25], row[25:]
    less_opponents = opponents[18 - 3:18 + 3 + 1]
    return np.concatenate([not_opponents, less_opponents])


# Smooth out a steering column.
def smooth_steering(column):
    # for i in range(len(column)):
    #     val = column[i]
    #     if val > 0:
    #         print(val)
    return np.convolve(column, (0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.3), "same")


def simplify_all(in_folder, out_folder):
    for path in glob.glob(os.path.join(".", in_folder, "*.csv")):
        data = np.genfromtxt(path, skip_header=True, delimiter=",", dtype=float)
        simplified_row_len = len(simplify_row(data[0]))
        simplified_data = np.empty((data.shape[0], simplified_row_len))
        for i in range(data.shape[0]):
            simplified_data[i] = simplify_row(data[i])
        filename = os.path.basename(path)
        np.savetxt(os.path.join(".", out_folder, filename),
                   simplified_data, header=",", delimiter=",")


def smooth_all(in_folder, out_folder):
    for path in glob.glob(os.path.join(".", in_folder, "*.csv")):
        data = np.genfromtxt(path, skip_header=True, delimiter=",", dtype=float)
        data[:, 2] = smooth_steering(data[:, 2])
        print(data[:, 2])
        filename = os.path.basename(path)
        np.savetxt(os.path.join(".", out_folder, filename), data, header=",",
                   delimiter=",")


if __name__ == "__main__":
    in_folder = "overtake_data"
    simp_folder = "simp_overtake_data"
    simp_smooth_folder = "simp_smooth_overtake_data"
    for folder in [in_folder, simp_folder, simp_smooth_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    simplify_all(in_folder, simp_folder)
    smooth_all(simp_folder, simp_smooth_folder)
