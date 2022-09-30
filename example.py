
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from potential_learner import PotentialLearner

# Set parameters
file = 'exampledata.h5'
num_data = 1000
parameters = {
    'dt': 1e6,    # Time step in nanoseconds
    'kT': 4.114,  # Temperature in kT
}

# Load data
h5 = h5py.File(file, 'r')
data = h5['data'][...]
h5.close()
data = data[:num_data]

# Run analysis
MAP = PotentialLearner.learn_potential(data, parameters=parameters)
PotentialLearner.plot_variables(data, MAP)

print("Done")
