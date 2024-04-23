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
