# data_visualization/data_visualization.py

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime


def load_data(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def plot_data(data, baseline_data=None, titles=None):
    if titles is None:
        titles = ['Training Errors', 'Training Accuracies', 'Test Errors', 'Test Accuracies']

    # 获取当前时间并格式化为字符串
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join('results', folder_name)
    # 创建以当前时间命名的文件夹
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for i, data_array in enumerate(data):
        plt.figure(figsize=(10, 5))
        plt.plot(data_array, 'g-', label='Our Algorithm')

        if baseline_data is not None:
            plt.plot(baseline_data[i], 'b-', label='MTFL-FedAdam')

        plt.title(titles[i])
        plt.xlabel('Epochs')
        plt.ylabel('Values')
        plt.legend()

        # 为每个图像创建一个文件名
        file_name = os.path.join(folder_name, f"figure_{i}.png")
        # 使用plt.savefig函数保存图像
        plt.savefig(file_name)

        plt.show()


def plot_from_file(file_name, baseline_file_name=None):
    data = load_data(file_name)

    if baseline_file_name:
        baseline_data = load_data(baseline_file_name)
    else:
        baseline_data = None

    plot_data(data, baseline_data=baseline_data)
