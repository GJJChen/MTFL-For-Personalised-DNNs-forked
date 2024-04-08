# data_visualization/data_visualization.py

import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_data(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def plot_data(data, baseline_data=None, titles=None):
    if titles is None:
        titles = ['Training Errors', 'Training Accuracies', 'Test Errors', 'Test Accuracies']

    for i, data_array in enumerate(data):
        plt.figure(figsize=(10, 5))
        plt.plot(data_array, 'g-', label='Current Data')

        if baseline_data is not None:
            plt.plot(baseline_data[i], 'b-', label='Baseline Data')

        plt.title(titles[i])
        plt.xlabel('Epochs')
        plt.ylabel('Values')
        plt.legend()
        plt.show()


def plot_from_file(file_name, baseline_file_name=None):
    data = load_data(file_name)

    if baseline_file_name:
        baseline_data = load_data(baseline_file_name)
    else:
        baseline_data = None

    plot_data(data, baseline_data=baseline_data)
