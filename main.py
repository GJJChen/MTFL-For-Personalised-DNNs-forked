import datetime
import os

# required for pytorch deterministic GPU behaviour
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import pickle
import torch
import time
from data_utils import *
from models import *
from optimisers import *
import argparse
from sys import argv
from fl_algs import *
from data_visualization.data_visualization import plot_from_file
from data_visualization import data_visualization as dv


def get_fname(a):
    """
    Args:
        - a: (argparse.Namespace) command-line arguments
        
    Returns:
        Underscore-separated str ending with '.pkl', containing items in args.
    """
    fname = '_'.join([k + '-' + str(v) for (k, v) in vars(a).items()
                      if not v is None])
    return fname + '.pkl'


def save_data(data, fname):
    """
    Saves data in pickle format.
    
    Args:
        - data:  (object)   to save 
        - fname: (str)      file path to save to 
    """
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def any_in_list(x, y):
    """
    Args:
        - x: (iterable) 
        - y: (iterable) 
    
    Returns:
        True if any items in x are in y.
    """
    return any(x_i in y for x_i in x)


def parse_args():
    """
    Details for the experiment to run are passed via the command line. Some 
    experiment settings require specific arguments to be passed (e.g. the 
    different FL algorithms require different hyperparameters). 
    
    Returns:
        argparse.Namespace of parsed arguments. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-dset', required=True, choices=['mnist', 'cifar10'],
                        help='Federated dataset', default='mnist')
    parser.add_argument('-alg', required=True, help='Federated optimiser',
                        choices=['fedavg', 'fedavg-adam', 'fedadam',
                                 'pfedme', 'perfedavg'],
                        default='fedavg')
    parser.add_argument('-C', required=True, type=float,
                        help='Fraction of clients selected per round', default=0.5)
    parser.add_argument('-B', required=True, type=int, help='Client batch size', default=20)
    parser.add_argument('-T', required=True, type=int, help='Total rounds', default=500)
    parser.add_argument('-E', required=True, type=int, help='Client num epochs', default=1)
    parser.add_argument('-device', required=True, choices=['cpu', 'gpu'],
                        help='Training occurs on this device', default='gpu')
    parser.add_argument('-W', required=True, type=int,
                        help='Total workers to split data across', default=200)
    parser.add_argument('-seed', required=True, type=int, help='Random seed', default=0)
    parser.add_argument('-lr', required=True, type=float,
                        help='Client learning rate', default=0.1)
    parser.add_argument('-noisy_frac', required=True, type=float,
                        help='Fraction of noisy clients', default=0.0)

    # specific arguments for different FL algorithms
    # argv 是一个包含命令行参数的列表，其中 argv[0] 是脚本的名称，而 argv[1:] 包含了传递给脚本的其余命令行参数。

    # 对于"fedavg"、"fedavg-adam"和"fedadam"算法，需要提供 -bn_private 参数，该参数用于指定要保留为私有的参数。
    if any_in_list(['fedavg', 'fedavg-adam', 'fedadam'], argv):
        parser.add_argument('-bn_private', choices=['usyb', 'us', 'yb', 'none'],
                            required=True,
                            help='Patch parameters to keep private',
                            default='none')
        parser.add_argument('-multi_gates', type=bool, help='If use the multi-gates', default=False, )

    # 对于"fedadam"算法，需要提供 -server_lr 参数，该参数用于指定服务器的学习率。
    if any_in_list(['fedadam'], argv):
        parser.add_argument('-server_lr', required=True, type=float,
                            help='Server learning rate')

    # 对于"perfedavg"和"pfedme"算法，需要提供 -beta 参数，参数β在pFedMe算法中用于控制全局模型的更新。
    # 当β=1时，它执行类似于FedAvg的模型平均，将所有客户端的本地模型进行简单平均
    # 当β的值不为1时，引入了一些个性化的全局模型更新
    if any_in_list(['perfedavg', 'pfedme'], argv):
        parser.add_argument('-beta', required=True, type=float,
                            help='PerFedAvg/pFedMe beta parameter')

    # 对于"pfedme"算法，需要提供 -lamda 参数，λ是一个正则化参数，用于控制全局模型 w 与个性化模型 θi 之间的关系强度。
    # 大 λ（λ增大）：可以使得不可靠数据的客户端从丰富的数据聚合中受益，因为它强调全局模型 w 对个性化模型的控制。
    # 小 λ（λ减小）：有助于拥有足够有用数据的客户端更加强调个性化。全局模型对个性化模型的控制较小，使得客户端更容易保留本地数据的特征。
    if 'pfedme' in argv:
        parser.add_argument('-lamda', required=True, type=float,
                            help='pFedMe lambda parameter')

    # β1、β2为Adam的一阶矩估计和二阶矩估计的指数衰减率，推荐值分别为0.9和0.999
    # ε通常设置为一个很小的数，如 1e-8。目的是防止出现除以0的错误
    if any_in_list(['fedavg-adam', 'fedadam'], argv):
        parser.add_argument('-beta1', required=True, type=float,
                            help='Only required for FedAdam, 0 <= beta1 < 1')
        parser.add_argument('-beta2', required=True, type=float,
                            help='Only required for FedAdam, 0 <= beta2 < 1')
        parser.add_argument('-epsilon', required=True, type=float,
                            help='Only required for FedAdam, 0 < epsilon << 1')

    args = parser.parse_args()

    return args


def simplify_filename(filename):
    # 通用地去掉文件名的扩展名
    filename, suffix = os.path.splitext(filename)
    # 定义要保留的关键字列表
    keep_keywords = ['B-', 'W-', 'lr-', 'bn', 'private-', 'multi', 'gates-']

    # 将文件名以下划线分割成多个部分
    parts = filename.split('_')

    # 初始化简化后的文件名列表
    simplified_filename = []

    # 遍历每个部分，检查是否包含关键字
    for part in parts:
        # 特别处理 'dset-' 和 'alg-' 前缀
        if 'dset-' in part or 'alg-' in part:
            # 提取并保留 'dset-' 或 'alg-' 后的值
            simplified_filename.append(part.split('-')[-1])
        elif any(keyword in part for keyword in keep_keywords):
            # 对其他关键字进行检查，并保留含有关键字的部分
            simplified_filename.append(part)

    # 使用下划线连接所有选中的部分，并添加'.pkl'扩展名
    simplified_filename = '_'.join(simplified_filename) + suffix

    return simplified_filename


def main():
    """
    Run experiment specified by command-line args.
    """

    args = parse_args()

    # 设置随机种子以进行确定性计算
    version = torch.__version__
    if version < '1.8.0':
        torch.set_deterministic(True)  # 设置为确定性计算模式，在相同的输入下，PyTorch 操作将产生相同的输出
    else:
        torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设置设备
    device = torch.device('cuda:0' if args.device == 'gpu' else 'cpu')

    # load data 
    print('Loading data...')
    if args.dset == 'mnist':
        dataset_name = 'MNIST'
        train, test = load_mnist('./MNIST_data', args.W, iid=False,
                                 user_test=True)
        if not args.multi_gates:
            model = MNISTModel(device).to(device)
        else:
            model = MNISTModel_MultiGates(device).to(device)

        noise_std = 3.0
        steps_per_E = int(np.round(60000 / (args.W * args.B)))  # 每轮本地训练的步数 = 总样本数 / (客户端数 * 客户端批量大小)

    else:
        dataset_name = 'CIFAR10'
        train, test = load_cifar('./CIFAR10_data', args.W,
                                 iid=False, user_test=True)
        if hasattr(args, 'multi_gates'):
            if not args.multi_gates:
                model = CIFAR10Model(device).to(device)
            else:
                model = CIFAR10Model_MultiGates(device).to(device)
        else:
            model = CIFAR10Model(device).to(device)
        noise_std = 0.2
        steps_per_E = int(np.round(50000 / (args.W * args.B)))

    # add noise to data
    noisy_imgs, noisy_idxs = add_noise_to_frac(train[0], args.noisy_frac,
                                               noise_std)  # 向数据集中的一部分添加噪声
    train = (noisy_imgs, train[1])  # 噪声图像 + 原始标签

    # convert to pytorch tensors
    feeders = [PyTorchDataFeeder(x, torch.float32, y, 'long', device)
               for (x, y) in zip(train[0], train[1])]  # x 是图像，y 是标签，train[0]是一个含W个元素的列表，每个元素为分配给客户端的数据
    test_data = ([to_tensor(x, device, torch.float32) for x in test[0]],
                 [to_tensor(y, device, 'long') for y in test[1]])

    # miscellaneous settings
    fname = get_fname(args)  # 生成一个文件名，构建一个以下划线分隔的字符串，以 '.pkl' 结尾
    M = int(args.W * args.C)  # 每轮选择的客户端数 = 总客户端数 * 选择比例
    K = steps_per_E * args.E  # 本地训练的总步数 = 每轮本地训练的步数 * 客户端训练轮数
    str_to_bn_setting = {'usyb': 0, 'yb': 1, 'us': 2, 'none': 3}
    if args.alg in ['fedavg', 'fedavg-adam', 'fedadam']:
        bn_setting = str_to_bn_setting[args.bn_private]  # 转字符参数换为数字参数

    # final check
    # 检查 baseline_file 是否存在
    baseline_file_name = 'MNIST_MTFL-FedAdam.pkl'
    baseline_file_name = os.path.join('results', 'baseline_data', baseline_file_name)
    if not os.path.exists(baseline_file_name):
        print(f"The baseline file {baseline_file_name} does not exist. Not specifying baseline_file_name.")
        baseline_file_name = None

    # run experiment
    print('Starting experiment...')
    if args.alg == 'fedavg':
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)
        data = run_fedavg(feeders, test_data, model, client_optim, args.T, M,
                          K, args.B, bn_setting=bn_setting,
                          noisy_idxs=noisy_idxs)

    elif args.alg == 'fedavg-adam':
        client_optim = ClientAdam(model.parameters(), lr=args.lr,
                                  betas=(args.beta1, args.beta2),
                                  eps=args.epsilon)
        model.set_optim(client_optim)
        data = run_fedavg(feeders, test_data, model, client_optim, args.T, M,
                          K, args.B, bn_setting=bn_setting,
                          noisy_idxs=noisy_idxs)

    elif args.alg == 'fedadam':
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)
        server_optim = ServerAdam(model.get_params(), args.server_lr,
                                  args.beta1, args.beta2, args.epsilon)
        data = run_fedavg_google(feeders, test_data, model,
                                 server_optim, args.T, M,
                                 K, args.B,
                                 bn_setting=bn_setting,
                                 noisy_idxs=noisy_idxs)

    elif args.alg == 'pfedme':
        client_optim = pFedMeOptimizer(model.parameters(), device,
                                       lr=args.lr, lamda=args.lamda)
        model.set_optim(client_optim, init_optim=False)
        data = run_pFedMe(feeders, test_data, model, args.T, M, K=1, B=args.B,
                          R=K, lamda=args.lamda, eta=args.lr,
                          beta=args.beta, noisy_idxs=noisy_idxs)

    elif args.alg == 'perfedavg':
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim, init_optim=False)
        data = run_per_fedavg(feeders, test_data, model, args.beta, args.T,
                              M, K, args.B, noisy_idxs=noisy_idxs)

    # 保存数据
    # 获取当前时间并格式化为字符串
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join('results', folder_name)

    # 创建以当前时间命名的文件夹
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    save_dir = os.path.join(folder_name, fname)
    save_data(data, save_dir)
    print('Data saved to: {}'.format(save_dir))

    # plot_from_file(save_dir, baseline_file_name, folder_name)

    # 根据数据集将文件名简化为更易读的格式
    simplified_fname = simplify_filename(fname)

    # 将数据复制到results/ComparisonExperiment的对应文件夹
    save_dir = os.path.join('results', 'ComparisonExperiment', dataset_name, simplified_fname)
    save_data(data, save_dir)
    dv.plot_all(os.path.join('results', 'ComparisonExperiment', dataset_name), dv.load_data)


if __name__ == '__main__':
    main()
