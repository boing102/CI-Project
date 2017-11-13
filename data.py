import numpy as np
import glob

# Collect & pre-process training data.
# See main for example usage.


# Training data from all files concatenated.
def all_data():
    data = []
    for filepath in glob.glob("./train_data/*.csv"):
        data.append(np.genfromtxt(filepath, skip_header=True, dtype=float,
                                  delimiter=",", skip_footer=True))
    return np.concatenate(data)


# Split data into training, validation and testing data, by given ratio.
def split_data(data, tr, va, te):
    sum_ = tr + va + te
    tr_end = int((tr / sum_) * len(data))
    va_end = int(((tr + va) / sum_) * len(data))
    return (data[:tr_end], data[tr_end + 1:va_end], data[va_end + 1:])


# Convert raw training data to a tuple (x, y). Where x is the sensor input and
# y the desired output.
def x_y(data):
    return (data[:, 3:], data[:, :3])


if __name__ == "__main__":
    all_ = all_data()
    print(all_.shape)
    tr, va, te = split_data(all_, 4, 1, 1)
    print(tr.shape)
    print(va.shape)
    print(te.shape)
    x, y = x_y(tr)
    print(x.shape)
    print(y.shape)
