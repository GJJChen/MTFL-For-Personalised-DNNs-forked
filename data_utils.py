import gzip
import numpy as np
import torch
import pickle


class PyTorchDataFeeder():
    """
    Contains data as torch.tensors. Allows easy retrieval of data batches and
    in-built data shuffling.
    """

    def __init__(self, x, x_dtype, y, y_dtype, device,
                 cast_device=None, transform=None):
        """
        Return a new data feeder with copies of the input x and y data. Data is 
        stored on device. If the intended model to use with the input data is 
        on another device, then cast_device can be passed. Batch data will be 
        sent to this device before being returned by next_batch. If transform is
        passed, the function will be applied to x data returned by next_batch.
        
        Args:
        - x:            x data to store
        - x_dtype:      torch.dtype or 'long' 
        - y:            y data to store 
        - y_dtype:      torch.dtype or 'long'
        - device:       torch.device to store data on 
        - cast_device:  data from next_batch is sent to this torch.device
        - transform:    function to apply to x data from next_batch
        """
        if x_dtype == 'long':
            self.x = torch.tensor(x, device=device,
                                  requires_grad=False,
                                  dtype=torch.int32).long()
        else:
            self.x = torch.tensor(x, device=device,
                                  requires_grad=False,
                                  dtype=x_dtype)

        if y_dtype == 'long':
            self.y = torch.tensor(y, device=device,
                                  requires_grad=False,
                                  dtype=torch.int32).long()
        else:
            self.y = torch.tensor(y, device=device,
                                  requires_grad=False,
                                  dtype=y_dtype)

        self.idx = 0
        self.n_samples = x.shape[0]
        self.cast_device = cast_device
        self.transform = transform
        self.shuffle_data()

    def shuffle_data(self):
        """
        Co-shuffle x and y data.
        """
        ord = torch.randperm(self.n_samples)
        self.x = self.x[ord]
        self.y = self.y[ord]

    def next_batch(self, B):
        """
        Return batch of data If B = -1, the all data is returned. Otherwise, a 
        batch of size B is returned. If the end of the local data is reached, 
        the contained data is shuffled and the internal counter starts from 0.
        
        Args:
        - B:    size of batch to return
        
        Returns (x, y) tuple of torch.tensors. Tensors are placed on cast_device
                if this is not None, else device.
        """
        if B == -1:
            x = self.x
            y = self.y
            self.shuffle_data()

        elif self.idx + B > self.n_samples:
            # if batch wraps around to start, add some samples from the start
            # 如果当前批次结束超过了数据集的末尾，需要特殊处理。
            # 计算需要从数据集开头补充的额外样本数量 extra。
            extra = (self.idx + B) - self.n_samples
            # 将当前批次的剩余部分 self.x[self.idx:] 和前面补充的样本 self.x[:extra] 连接起来，形成新的 x。
            x = torch.cat((self.x[self.idx:], self.x[:extra]))
            y = torch.cat((self.y[self.idx:], self.y[:extra]))
            self.shuffle_data()
            self.idx = extra

        else:
            x = self.x[self.idx:self.idx + B]
            y = self.y[self.idx:self.idx + B]
            self.idx += B

        if not self.cast_device is None:
            x = x.to(self.cast_device)
            y = y.to(self.cast_device)

        if not self.transform is None:
            x = self.transform(x)

        return x, y


def load_mnist(data_dir, W, iid, user_test=False):
    """
    Load the MNIST dataset. The folder specified by data_dir should contain the
    standard MNIST files from (yann.lecun.com/exdb/mnist/). 

    Args:
        - data_dir:  (str)  path to data folder
        - W:         (int)  number of workers to split dataset into
        - iid:       (bool) iid (True) or non-iid shard-based split (False)
        - user_test: (bool) split test data into users
        
    Returns:
        Tuple containing ((x_train, y_train), (x_test, y_test)). The training 
        variables are both lists of length W, each element being a 2D numpy 
        array. If user_test is True, the test variables will also be lists, 
        otherwise the returned test values are just 2D numpy arrays.
    """
    train_x_fname = data_dir + '/train-images-idx3-ubyte.gz'
    train_y_fname = data_dir + '/train-labels-idx1-ubyte.gz'
    test_x_fname = data_dir + '/t10k-images-idx3-ubyte.gz'
    test_y_fname = data_dir + '/t10k-labels-idx1-ubyte.gz'

    # load MNIST files
    with gzip.open(train_x_fname) as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        x_train = x_train.astype(np.float32) / 255.0

    with gzip.open(train_y_fname) as f:
        y_train = np.copy(np.frombuffer(f.read(), np.uint8, offset=8))

    with gzip.open(test_x_fname) as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        x_test = x_test.astype(np.float32) / 255.0

    with gzip.open(test_y_fname) as f:
        y_test = np.copy(np.frombuffer(f.read(), np.uint8, offset=8))

    # split into iid/non-iid and users
    if iid:
        x_train, y_train = co_shuffle_split(x_train, y_train, W)
        if user_test:
            x_test, y_test = co_shuffle_split(x_test, y_test, W)

    else:
        x_train, y_train, assign = shard_split(x_train, y_train, W, W * 2)
        if user_test:
            x_test, y_test, _ = shard_split(x_test, y_test, W, W * 2, assign)

    return (x_train, y_train), (x_test, y_test)


def load_cifar(data_dir, W, iid, user_test=False):
    """
    Load the CIFAR dataset. The folder specified by data_dir should contain the
    python version pickle files from (cs.toronto.edu/~kriz/cifar.html). 

    Args:
        - data_dir:  (str)  path to data folder
        - W:         (int)  number of workers to split dataset into
        - iid:       (bool) iid (True) or non-iid shard-based split (False)
        - user_test: (bool) split test data into users
        
    Returns:
        Tuple containing ((x_train, y_train), (x_test, y_test)). The training 
        variables are both lists of length W, each element being a 4D numpy 
        array. If user_test is True, the test variables will also be lists, 
        otherwise the returned test values are just 4D numpy arrays.
    """
    fnames = ['/data_batch_1',
              '/data_batch_2',
              '/data_batch_3',
              '/data_batch_4',
              '/data_batch_5']

    # create big arrays to store all CIFAR train data, load and assign
    x_train = np.zeros((50000, 32, 32, 3), dtype=np.float32)
    y_train = np.zeros((50000), dtype=np.int32)

    for i in range(len(fnames)):
        with open(data_dir + fnames[i], 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')

        images = data_dict[b'data'].reshape((10000, 32, 32, 3), order='F')
        images = np.rot90(images, k=3, axes=(1, 2)) / 255.0
        labels = np.array(data_dict[b'labels'])

        x_train[i * 10000:(i + 1) * 10000, :, :, :] = images
        y_train[i * 10000:(i + 1) * 10000] = labels

    # load test set data
    with open(data_dir + '/test_batch', 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    x_test = data_dict[b'data'].reshape((10000, 32, 32, 3), order='F')
    x_test = np.rot90(x_test, k=3, axes=(1, 2)) / 255.0
    y_test = np.array(data_dict[b'labels'])

    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))

    # split into iid/non-iid and users
    if iid:
        x_train, y_train = co_shuffle_split(x_train, y_train, W)
        if user_test:
            x_test, y_test = co_shuffle_split(x_test, y_test, W)

    else:
        x_train, y_train, assign = shard_split_new(x_train, y_train, W, W * 2)
        if user_test:
            x_test, y_test, _ = shard_split_new(x_test, y_test, W, W * 2, assignment=assign)

    return (x_train, y_train), (x_test, y_test)


def add_noise_to_frac(xs, frac, std):
    """
    Random 0-mean gaussian noise with given std will be added to (frac*len(xs))
    of the arrays in xs. Noisy values are clipped between 0-1. 
    
    Args:
        - xs:   (list)      containing numpy ndarrays to add noise to
        - frac: (float) 0   <= frac <= 1, fraction of xs to add noise to
        - std:  (float)     standard deviation of noise. 
        
    Returns:
        Tuple containing (noisy copy of xs, indexes of noisy vals)
    """
    idxs = np.random.choice(len(xs), int(len(xs) * frac), replace=False)

    new_xs = []
    for i in range(len(xs)):
        if i in idxs:
            noisy = xs[i] + np.random.normal(0.0, std, size=xs[i].shape)
            new_xs.append(np.clip(noisy, 0.0, 1.0))
        else:
            new_xs.append(np.copy(xs[i]))

    return new_xs, idxs


def co_shuffle_split(x, y, W):
    """
    Shuffle x and y using the same random order, split into W parts.
    
    Args:
        - x: (np.ndarray) samples
        - y: (np.ndarray) labels
        - W: (int)        num parts to split into
            
    Returns:
        Tuple containing (list of x arrays, list of y arrays)
    """
    order = np.random.permutation(x.shape[0])
    x_split = np.array_split(x[order], W)
    y_split = np.array_split(y[order], W)

    return x_split, y_split


def shard_split(x, y, W, n_shards, assignment=None):
    """
    Split x and y into W parts. Arrays are sorted according to classes in y,
    split into n_shards. If assignment is None, each W is assigned random
    shards, otherwise, the passed assignment is used. This function therefore
    creates a non-iid split based on classes.

    Args:
        - x:         (np.ndarray) samples
        - y:         (np.ndarray) labels
        - W:         (int)        num parts to split into
        - n_shards   (int)        num shards per W
        - assignment (np.array)   pre-determined shard assingment, or None

    Returns:
        Tuple containing (list of x arrays, list of y arrays, assingment), where
        assignment is created at random if passed assignment is None, otherwise
        just passed back out.
    """
    order = np.argsort(y)
    x_sorted = x[order]
    y_sorted = y[order]

    # split data into shards of (mostly) the same index
    x_shards = np.array_split(x_sorted, n_shards)
    y_shards = np.array_split(y_sorted, n_shards)

    if assignment is None:
        assignment = np.array_split(np.random.permutation(n_shards), W)

    x_sharded = []
    y_sharded = []

    # assign each worker two shards from the random assignment
    for w in range(W):
        x_sharded.append(np.concatenate([x_shards[i] for i in assignment[w]]))
        y_sharded.append(np.concatenate([y_shards[i] for i in assignment[w]]))

    return x_sharded, y_sharded, assignment


def shard_split_new(x, y, W, n_shards, unique_clients_fraction=0.2, assignment=None):
    """
    Split x and y into W parts with specified proportions for unique data clients.
    The data for non-unique clients is non-IID and based on class labels.

    Args:
        - x:                     (np.ndarray) samples
        - y:                     (np.ndarray) labels
        - W:                     (int)        num parts to split into
        - n_shards:              (int)        num shards per W
        - unique_clients_fraction: (float)    fraction of clients that will have unique data

    Returns:
        Tuple containing (list of x arrays, list of y arrays), representing the data
        for each client.
    """
    # Determine the number of unique clients
    n_unique_clients = int(W * unique_clients_fraction)
    n_regular_clients = W - n_unique_clients
    n_regular_shards = n_shards - 2 * n_unique_clients
    samples_per_client = len(x) // W

    # Shuffle the data for unique clients
    unique_indices = np.random.permutation(len(x))[:n_unique_clients * samples_per_client]

    # Initialize lists to hold the client data
    x_clients = []
    y_clients = []

    # Assign unique data to unique clients
    for i in range(n_unique_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        x_clients.append(x[unique_indices[start_idx:end_idx]])
        y_clients.append(y[unique_indices[start_idx:end_idx]])

    # Sort remaining data by labels for non-IID distribution
    order = np.argsort(y)
    x_sorted = x[order]
    y_sorted = y[order]

    # Split sorted data into shards
    x_shards = np.array_split(x_sorted, n_regular_shards)
    y_shards = np.array_split(y_sorted, n_regular_shards)

    if assignment is None:
        assignment = np.array_split(np.random.permutation(n_regular_shards), n_regular_clients)

    # assign each worker two shards from the random assignment
    for w in range(n_regular_clients):
        x_clients.append(np.concatenate([x_shards[i] for i in assignment[w]]))
        y_clients.append(np.concatenate([y_shards[i] for i in assignment[w]]))

    return x_clients, y_clients, assignment


def to_tensor(x, device, dtype):
    """
    Convert Numpy array to torch.tensor.
    
    Args:
    - x:        (np.ndarray)   array to convert
    - device:   (torch.device) to place tensor on
    - dtype:    (torch.dtype)  or 'long' to convert to pytorch long
    """
    if dtype == 'long':
        return torch.tensor(x, device=device,
                            requires_grad=False,
                            dtype=torch.int32).long()
    else:
        return torch.tensor(x, device=device,
                            requires_grad=False,
                            dtype=dtype)
