# CI-Project

## Files

The driver we use is in `my_driver.py`.

It is trained with `nn-training.py`.

The NN topology is optimized with a GA in `ga.py`.

A `Driver` capable of being steered with manual control
and recording data is in `data_collection_driver.py` which
can be started with `start-data-collection.sh`.

`fix_data.py` removes and smoothes out collected data.

`ann-0.py` is an unused simply self-implemented NN.

## Folders

Results of the GA are in `ga-results/`.

Trained NN models are in `models/`.

`train_data/` and `overtake_data/` contain data used to train the
racing and overtaking NNs respectively.

`output/` contains result output from races.
