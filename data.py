import numpy as np
import glob

# Example usage:
#
#   tuples(all_data())


# Training data from all files concatenated.
def all_data():
    data = []
    for filepath in glob.glob("./train_data/*.csv"):
        data.append(np.genfromtxt(filepath, skip_header=True, dtype=float,
                                  delimiter=",", skip_footer=True))
    return np.concatenate(data)


# Convert raw training data to a list of tuples (x, y). Where x is the sensor
# input and y the desired output.
def tuples(data):
    return (data[:, :3], data[:, 3:])


if __name__ == "__main__":
    all_ = all_data()
    print(all_.shape)
    pretty = tuples(all_)
    print(pretty[0].shape)
    print(pretty[1].shape)
