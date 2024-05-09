# data_visualization/data_visualization.py

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime


def load_data(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def plot_data(data, baseline_data=None, titles=None, folder_name=None):  # 添加 folder_name 参数
    if titles is None:
        titles = ['Training Errors', 'Training Accuracies', 'Test Errors', 'Test Accuracies', 'local_gate_weights']

    # 创建子文件夹 fig
    fig_folder = os.path.join(folder_name, 'fig')
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    for i, data_array in enumerate(data):
        plt.figure(figsize=(10, 5))
        plt.plot(data_array, 'g-', label='MultiGates Algorithm')

        if baseline_data is not None:
            plt.plot(baseline_data[i], 'b-', label='MTFL-FedAdam')

        plt.title(titles[i])
        plt.xlabel('Epochs')
        plt.ylabel('Values')
        plt.legend()

        # 为每个图像创建一个文件名
        file_name = os.path.join(fig_folder, f"figure_{i}.png")
        # 使用plt.savefig函数保存图像
        plt.savefig(file_name)

        plt.show()


def plot_from_file(file_name, baseline_file_name=None, folder_name=None):  # 添加 folder_name 参数
    data = load_data(file_name)

    if baseline_file_name:
        baseline_data = load_data(baseline_file_name)
    else:
        baseline_data = None

    plot_data(data, baseline_data=baseline_data, folder_name=folder_name)  # 添加 folder_name 参数


def plot_metric(metric_name, metric_data, save_path):
    plt.figure(figsize=(10, 5))
    for file_name, values in metric_data:
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=os.path.splitext(file_name)[0])
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.split('_')[1].title())
    plt.legend()
    plt.grid(True)
    # Save the plot in the designated folder with a unique name
    plot_filename = f"{metric_name}.png"
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.show()
    plt.close()


def plot_all(directory, data_handler):
    # Prepare a dictionary to store the data for plotting
    plot_data = {
        'training_errors': [],
        'training_accuracies': [],
        'test_errors': [],
        'test_accuracies': []
    }

    # Load data from each .pkl file
    for file_name in os.listdir(directory):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(directory, file_name)
            # Unpack the data from the loaded pickle file using the provided data handler
            training_errors, training_accuracies, test_errors, test_accuracies = data_handler(file_path)

            # Append the data for plotting
            plot_data['training_errors'].append((file_name, training_errors))
            plot_data['training_accuracies'].append((file_name, training_accuracies))
            plot_data['test_errors'].append((file_name, test_errors))
            plot_data['test_accuracies'].append((file_name, test_accuracies))

    # Current date and time as folder name
    now = datetime.datetime.now()
    folder_name = now.strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(directory, 'fig', folder_name)
    os.makedirs(save_path, exist_ok=True)

    # Plot each metric
    for metric_name, metric_data in plot_data.items():
        plot_metric(metric_name, metric_data, save_path)
