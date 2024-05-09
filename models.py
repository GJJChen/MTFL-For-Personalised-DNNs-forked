import torch
import numpy as np
import operator
import numbers


# from torch.profiler import profile, record_function, ProfilerActivity


class FLModel(torch.nn.Module):
    """
    Has methods to easily allow setting of model parameters, performing training 
    steps, and retrieving model parameters.
    """

    def __init__(self, device):
        """
        Args:
            - device: (torch.device) to place model on
        """
        super(FLModel, self).__init__()
        self.optim = None
        self.device = device
        self.loss_fn = None
        self.bn_layers = []  # any model BN layers must be added to this
        self.local_gates = []  # any model gate networks must be added to this

    def set_optim(self, optim, init_optim=True):
        """
        Set the optimizer that this model will perform SGD with.
        
        Args:
            - optim:     (torch.optim) that model will perform SGD steps with
            - init_optim (bool)        whether to initialise optimiser params
        """
        self.optim = optim
        if init_optim:
            self.empty_step()

    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialise 
        optimiser parameters.
        """
        raise NotImplementedError()  # 子类中需重写实现

    def get_params(self):
        """
        Returns model values as NumpyModel. BN layer statistics (mu, sigma) are
        added as parameters at the end of the returned model.
        'self.parameters()' 返回模型中所有需要进行学习的参数
        """
        params = [np.copy(param.data.cpu().numpy()) for param in list(self.parameters())]
        for bn in self.bn_layers:
            params.append(np.copy(bn.running_mean.cpu().numpy()))  # Bn层的均值
            params.append(np.copy(bn.running_var.cpu().numpy()))  # Bn层的方差

        return NumpyModel(params)

    def get_bn_vals(self, setting=0):
        """
        Returns the parameters from BN layers. Setting can be one of the 
        following {0: usyb, 1: yb, 2: us, 3: none} to get different types of 
        parameters.
        0-3分别对应一种隐私化方式
        
        Args:
            - setting: (int) BN values to return 
            
        Returns:
            list of [np.ndarrays] containing BN parameters
        """
        if setting not in [0, 1, 2, 3]:
            raise ValueError('Setting must be in: {0, 1, 2, 3}')

        vals = []

        if setting == 3:
            return vals

        with torch.no_grad():
            # add gamma, beta
            if setting in [0, 1]:
                for bn in self.bn_layers:
                    vals.append(np.copy(bn.weight.cpu().numpy()))  # gamma
                    vals.append(np.copy(bn.bias.cpu().numpy()))  # beta

            # add mu, sigma
            if setting in [0, 2]:
                for bn in self.bn_layers:
                    vals.append(np.copy(bn.running_mean.cpu().numpy()))  # mu
                    vals.append(np.copy(bn.running_var.cpu().numpy()))  # sigma
        return vals

    def set_bn_vals(self, vals, setting=0):
        """
        Set the BN parameterss of the model. Setting can be one of the following
        {0: usyb, 1: yb, 2: us, 3: none}.
        
        Args:
            - vals:     (NumpyModel) new BN values to set
            - setting:  (int)        type of values to return
        """
        if setting not in [0, 1, 2, 3]:
            raise ValueError('Setting must be in: {0, 1, 2, 3}')

        if setting == 3:
            return

        with torch.no_grad():
            i = 0
            # set gamma, beta
            if setting in [0, 1]:
                for bn in self.bn_layers:
                    bn.weight.copy_(torch.tensor(vals[i]))
                    bn.bias.copy_(torch.tensor(vals[i + 1]))
                    i += 2

            # set mu, sigma
            if setting in [0, 2]:
                for bn in self.bn_layers:
                    bn.running_mean.copy_(torch.tensor(vals[i]))
                    bn.running_var.copy_(torch.tensor(vals[i + 1]))
                    i += 2

    def set_params(self, params):
        """
        Passed params should be in the order of the model layers, as returned by
        get_params(), with the BN layer statistics (mu, sigma) appended to the 
        end of the model.
        
        Args:
            - params:   (NumpyModel) to set model values with
        """
        i = 0
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.tensor(params[i]))
                i += 1

            # set mu, sigma
            for bn in self.bn_layers:
                bn.running_mean.copy_(torch.tensor(params[i]))
                bn.running_var.copy_(torch.tensor(params[i + 1]))
                i += 2

    def get_local_gate_vals(self):
        """
        Returns the parameters from gate networks.

        Returns:
            - list of torch.Tensor containing gate network parameters
        """
        return [gate.parameters() for gate in self.local_gates]

    def set_local_gate_vals(self, vals):
        """
        Set the gate parameters of the model.

        Args:
            - vals: list of torch.Tensor containing new gate parameters
        """
        with torch.no_grad():
            for gate, val in zip(self.local_gates, vals):
                for param, new_val in zip(gate.parameters(), val):
                    param.copy_(new_val)

    def forward(self, x):
        """
        Returns outputs of model given data x.
        
        Args:
            - x: (torch.tensor) must be on same device as model
            
        Returns:
            torch.tensor model outputs
        """
        raise NotImplementedError()

    def calc_acc(self, logits, y):
        """
        Calculate accuracy/performance metric of model.
        
        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values
            
        Returns:
            torch.tensor containing scalar value.
        """
        raise NotImplementedError()

    def train_step(self, x, y):
        """
        Perform one step of SGD using assigned optimizer.
        
        Args:
            - x: (torch.tensor) inputs
            - y: (torch.tensor) targets
        D
        Returns:
            tupe of floats (loss, acc) calculated during the training step.
        """
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.calc_acc(logits, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item(), acc

    def test(self, x, y, B):
        """
        Calculate error and accuracy of passed data using batches of size B.
        
        Args:
            - x: (torch.tensor) inputs
            - y: (torch.tensor) labels
            - B: (int)          batch size
        
        Returns: 
            tuple of floats (loss, acc) averaged over passed data.
        """
        self.eval()
        n_batches = int(np.ceil(x.shape[0] / B))
        loss = 0.0
        acc = 0.0

        with torch.no_grad():
            for b in range(n_batches):
                logits = self.forward(x[b * B:(b + 1) * B])
                loss += self.loss_fn(logits, y[b * B:(b + 1) * B]).item()
                acc += self.calc_acc(logits, y[b * B:(b + 1) * B])
        self.train()

        return loss / n_batches, acc / n_batches


class MNISTModel(FLModel):
    """
    2-hidden-layer fully connected model, 2 hidden layers with 200 units and a 
    BN layer. Categorical Cross Entropy loss.
    """

    def __init__(self, device):
        """
        Returns a new MNISTModelBN.
        
        Args:
            - device: (torch.device) to place model on
        """
        super(MNISTModel, self).__init__(device)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.fc0 = torch.nn.Linear(784, 200).to(device)
        self.relu0 = torch.nn.ReLU().to(device)

        self.fc1 = torch.nn.Linear(200, 200).to(device)
        self.relu1 = torch.nn.ReLU().to(device)

        self.out = torch.nn.Linear(200, 10).to(device)

        self.bn0 = torch.nn.BatchNorm1d(200).to(device)

        self.bn_layers = [self.bn0]

    def forward(self, x):
        """
        Returns outputs of model given data x.
        
        Args:
            - x: (torch.tensor) must be on same device as model
            
        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        a = self.bn0(self.relu0(self.fc0(x)))
        b = self.relu1(self.fc1(a))

        return self.out(b)

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.
        
        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values
            
        Returns:
            torch.tensor containing scalar value.
        """
        return (torch.argmax(logits, dim=1) == y).float().mean()

    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialse 
        optimiser parameters.
        """
        self.train_step(torch.zeros((2, 784),
                                    device=self.device,
                                    dtype=torch.float32),
                        torch.zeros((2),
                                    device=self.device,
                                    dtype=torch.int32).long())


class GateNetwork(torch.nn.Module):
    def __init__(self, input_size):
        super(GateNetwork, self).__init__()
        # 假设门控网络有一个简单的线性层
        self.fc_gate = torch.nn.Linear(input_size, 2)  # 输出2个权重

    def forward(self, x):
        gate_output = self.fc_gate(x)
        # 使用softmax激活函数来确保权重的总和为1
        gate_weights = torch.softmax(gate_output, dim=1)
        return gate_weights


class MNISTModel_MultiGates(FLModel):
    """
    2-hidden-layer fully connected model, 2 hidden layers with 200 units and a
    BN layer. 1 local gate and 1 global gate. Categorical Cross Entropy loss.
    """

    def __init__(self, device):
        """
        Returns a new MNISTModelBN.

        Args:
            - device: (torch.device) to place model on
        """
        super(MNISTModel_MultiGates, self).__init__(device)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.fc0 = torch.nn.Linear(784, 200).to(device)
        self.relu0 = torch.nn.ReLU().to(device)

        self.fc1 = torch.nn.Linear(200, 200).to(device)
        self.relu1 = torch.nn.ReLU().to(device)

        self.out = torch.nn.Linear(200, 10).to(device)

        self.local_bn0 = torch.nn.BatchNorm1d(200).to(device)
        self.global_bn0 = torch.nn.BatchNorm1d(200).to(device)

        self.bn_layers = [self.local_bn0]

        # multi-gate
        self.local_gate_network = GateNetwork(200).to(device)
        self.gates = [self.local_gate_network]

    def forward(self, x):
        """
        Returns outputs of model given data x.

        Args:
            - x: (torch.tensor) must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """

        """
        Multi-Gate
        """
        shared_output = self.relu0(self.fc0(x))

        # 局部和全局BN层的输出
        local_bn_output = self.local_bn0(shared_output)
        # global_bn_output = self.global_bn0(shared_output)
        # 计算门控权重
        gate_weights = self.local_gate_network(shared_output)
        # 根据权重聚合BN层的输出
        # aggregated_bn_output = gate_weights[:, 0:1] * local_bn_output + gate_weights[:, 1:2] * global_bn_output
        # 残差连接
        aggregated_bn_output = gate_weights[:, 0:1] * local_bn_output + gate_weights[:, 1:2] * shared_output

        expert_output = self.relu1(self.fc1(aggregated_bn_output))

        # 输出层
        final_output = self.out(expert_output)

        return final_output

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values

        Returns:
            torch.tensor containing scalar value.
        """
        return (torch.argmax(logits, dim=1) == y).float().mean()

    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialse
        optimiser parameters.
        """
        self.train_step(torch.zeros((2, 784),
                                    device=self.device,
                                    dtype=torch.float32),
                        torch.zeros(2,
                                    device=self.device,
                                    dtype=torch.int32).long())


class LocalResidualGate_Linear(torch.nn.Module):
    def __init__(self, channels):
        super(LocalResidualGate_Linear, self).__init__()
        self.conv1x1 = torch.nn.Conv2d(channels, channels, 1)
        self.gate_network = GateNetwork_CNN(channels)
        self.bn0 = torch.nn.BatchNorm2d(channels).to("cuda:0")

    def forward(self, x):
        residual = self.conv1x1(self.bn0(x))
        gate_weights = self.gate_network(x)
        return gate_weights[:, 0:1] * x + gate_weights[:, 1:2] * residual


class CIFAR10Model(FLModel):
    """
    Convolutional model with two (Conv -> ReLU -> MaxPool -> BN) blocks, and one 
    fully connected hidden layer. Categorical Cross Entropy loss.
    """

    def __init__(self, device):
        """
        Returns a new CIFAR10Model.
        
        Args:
            - device: (torch.device) to place model on
        """
        super(CIFAR10Model, self).__init__(device)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1).to(device)
        self.relu0 = torch.nn.ReLU().to(device)
        self.pool0 = torch.nn.MaxPool2d(2, 2).to(device)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1).to(device)
        self.relu1 = torch.nn.ReLU().to(device)
        self.pool1 = torch.nn.MaxPool2d(2, 2).to(device)

        self.flat = torch.nn.Flatten().to(device)
        self.fc0 = torch.nn.Linear(2304, 512).to(device)
        self.relu2 = torch.nn.ReLU().to(device)

        self.out = torch.nn.Linear(512, 10).to(device)

        self.bn0 = torch.nn.BatchNorm2d(32).to(device)
        self.bn1 = torch.nn.BatchNorm2d(64).to(device)

        self.bn_layers = [self.bn0, self.bn1]

    def forward(self, x):
        """
        Returns outputs of model given data x.
        
        Args:
            - x: (torch.tensor) must be on same device as model
            
        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        a = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        b = self.bn1(self.pool1(self.relu1(self.conv1(a))))
        c = self.relu2(self.fc0(self.flat(b)))

        return self.out(c)

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.
        
        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values
            
        Returns:
            torch.tensor containing scalar value.
        """
        return (torch.argmax(logits, dim=1) == y).float().mean()

    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialise 
        optimiser parameters.
        """
        self.train_step(torch.zeros((2, 3, 32, 32),
                                    device=self.device,
                                    dtype=torch.float32),
                        torch.zeros((2),
                                    device=self.device,
                                    dtype=torch.int32).long())


class GateLayer(torch.nn.Module):
    def __init__(self, in_channels):
        super(GateLayer, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, local_out, global_out):
        combined = torch.cat([local_out, global_out], dim=1)
        gate = torch.sigmoid(self.conv(combined))
        return gate * local_out + (1 - gate) * global_out


class GateNetwork_CNN(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super(GateNetwork_CNN, self).__init__()
        hidden_dim = max(channels // reduction, 8)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(channels, hidden_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1).view(-1, 2, 1, 1)


class LocalResidualGate(torch.nn.Module):
    def __init__(self, channels):
        super(LocalResidualGate, self).__init__()
        self.conv1x1 = torch.nn.Conv2d(channels, channels, 1)
        self.gate_network = GateNetwork_CNN(channels)
        self.bn0 = torch.nn.BatchNorm2d(channels).to("cuda:0")

    def forward(self, x):
        residual = self.conv1x1(x)
        gate_weights = self.gate_network(x)
        return gate_weights[:, 0:1] * self.bn0(x) + gate_weights[:, 1:2] * residual


class CIFAR10Model_MultiGates(FLModel):
    """
    Convolutional model with two (Conv -> ReLU -> MaxPool -> BN) blocks, and one
    fully connected hidden layer. Categorical Cross Entropy loss.
    """

    def __init__(self, device):
        """
        Returns a new CIFAR10Model.

        Args:
            - device: (torch.device) to place model on
        """
        super(CIFAR10Model_MultiGates, self).__init__(device)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1).to(device)
        self.relu0 = torch.nn.ReLU().to(device)
        self.pool0 = torch.nn.MaxPool2d(2, 2).to(device)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1).to(device)
        self.relu1 = torch.nn.ReLU().to(device)
        self.pool1 = torch.nn.MaxPool2d(2, 2).to(device)

        self.flat = torch.nn.Flatten().to(device)
        self.fc0 = torch.nn.Linear(2304, 512).to(device)
        self.relu2 = torch.nn.ReLU().to(device)

        self.out = torch.nn.Linear(512, 10).to(device)

        self.bn0 = torch.nn.BatchNorm2d(32).to(device)
        self.bn1 = torch.nn.BatchNorm2d(64).to(device)

        self.global_bn0 = torch.nn.BatchNorm2d(32).to(device)
        # self.global_bn1 = torch.nn.BatchNorm2d(64).to(device)

        self.gate_network0 = GateNetwork_CNN(32).to(device)
        # self.gate_network1 = GateNetwork_CNN(64).to(device)
        self.local_residual_gate0 = LocalResidualGate(32).to(device)
        self.local_residual_gate1 = LocalResidualGate(64).to(device)
        self.conv1x1_0 = torch.nn.Conv2d(64, 32, 1).to(device)
        # self.conv1x1_1 = torch.nn.Conv2d(128, 64, 1).to(device)
        self.bn_layers = [self.bn0, self.bn1]
        # self.gate_layers = [self.local_residual_gate0, self.local_residual_gate1]
        self.gate_layers = [self.gate_network0]

    def forward(self, x):
        """
        Returns outputs of model given data x.

        Args:
            - x: (torch.tensor) must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        shared_output0 = self.pool0(self.relu0(self.conv0(x)))

        # local_feat0 = self.local_residual_gate0(shared_output0)
        # b = self.bn1(self.pool1(self.relu1(self.conv1(local_feat0))))

        local_feat0 = self.bn0(shared_output0)
        global_feat0 = self.global_bn0(shared_output0)
        gate_weights0 = self.gate_network0(shared_output0)
        aggregated_feat0 = gate_weights0[:, 0:1] * local_feat0 + gate_weights0[:, 1:2] * global_feat0
        b = self.bn1(self.pool1(self.relu1(self.conv1(aggregated_feat0))))

        c = self.relu2(self.fc0(self.flat(b)))
        return self.out(c)

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values

        Returns:
            torch.tensor containing scalar value.
        """
        return (torch.argmax(logits, dim=1) == y).float().mean()

    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialise
        optimiser parameters.
        """
        self.train_step(torch.zeros((2, 3, 32, 32),
                                    device=self.device,
                                    dtype=torch.float32),
                        torch.zeros((2),
                                    device=self.device,
                                    dtype=torch.int32).long())

    def set_bn_params(self, params):
        self.bn0.load_state_dict(params[0])
        self.bn1.load_state_dict(params[1])

    def set_local_gate_vals(self, vals):
        for i, gate in enumerate(self.gate_layers):
            gate.load_state_dict(vals[i])

    def get_local_gate_vals(self):
        return [gate.state_dict() for gate in self.gate_layers]


class NumpyModel():
    """
    Allows easy operations on whole model of parameters using numpy arrays.
    """

    def __init__(self, params):
        """
        Return a NumpyModel using given params (values are not copied).
        使用给定的参数返回一个NumpyModel（不复制值，共享相同的内存）。
        Args:
            - params: (list) of numpy arrays representing model parameters.
            - params: (list) 表示模型参数的numpy数组。
        """
        self.params = params

    def copy(self):
        """
        Returns: 
            (NumpyModel) with all parameters copied from this model.
            返回的是复制值，不是共享内存
        """
        return NumpyModel([np.copy(p) for p in self.params])

    def zeros_like(self):
        """
        Returns:
            (NumpyModel) with all-0 values.
        """
        return NumpyModel([np.zeros_like(p) for p in self.params])

    def _op(self, other, f):
        """
        Return a new NumpyModel, where each parameter is computed using function
        f of this model's parameters and the other model's corresponding
        parameters/a constant value.
        
        Args:
            - other: (NumpyModel) or float/int 
            - f:     (function)   to apply
        """
        if np.isscalar(other):
            new_params = [f(p, other) for p in self.params]

        elif isinstance(other, NumpyModel):
            new_params = [f(p, o) for (p, o) in zip(self.params, other.params)]

        else:
            raise ValueError('Incompatible type for op: {}'.format(other))

        return NumpyModel(new_params)

    def abs(self):
        """
        Returns:
            (NumpyModel) with all absolute values.
        """
        return NumpyModel([np.absolute(p) for p in self.params])

    def __add__(self, other):
        """
        Args:
            - other: (NumpyModel) or float/int.
            
        Returns:
            (NumpyModel) of self + other, elementwise.
        """
        return self._op(other, operator.add)

    def __radd__(self, other):
        """
        Returns new NumpyModel with vals of (self + other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.add)

    def __sub__(self, other):
        """
        Returns new NumpyModel with vals of (self - other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.sub)

    def __mul__(self, other):
        """
        Returns new NumpyModel with vals of (self * other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.mul)

    def __rmul__(self, other):
        """
        Returns new NumpyModel with vals of (self * other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.mul)

    def __truediv__(self, other):
        """
        Returns new NumpyModel with vals of (self / other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.truediv)

    def __pow__(self, other):
        """
        Returns new NumpyModel with vals of (self ^ other).
        
        Args:
            - other: (NumpyModel) or float/int.
        """
        return self._op(other, operator.pow)

    def __getitem__(self, key):
        """
        Get parameter [key] of model.
        """
        return self.params[key]

    def __len__(self):
        """
        Return number of parameters in model.
        """
        return len(self.params)

    def __iter__(self):
        """
        Iterate over parameters of model.
        """
        for p in self.params:
            yield p
