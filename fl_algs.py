import numpy as np
import pickle
import torch
from progressbar import progressbar
from models import NumpyModel
import torch.nn.functional as F
from torch.nn.functional import kl_div


# from torch.profiler import profile, record_function, ProfilerActivity

def init_stats_arrays(T):
    """
    Returns:
        (tupe) of 4 numpy 0-filled float32 arrays of length T.
    """
    return tuple(np.zeros(T, dtype=np.float32) for i in range(4))


def run_fedavg_google(data_feeders, test_data, model, server_opt, T, M,
                      K, B, test_freq=1, bn_setting=0, noisy_idxs=[]):
    """
    Run the Adaptive Federated Optimization (AdaptiveFedOpt) algorithm from 
    'Adaptive Federated Optimization', Reddi et al., ICLR 2021. AdaptiveFedOpt 
    uses SGD on clients and a generic server optimizer to update the global 
    model each round. Runs T rounds of AdaptiveFedOpt, and returns the training 
    and test results.
    
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - server_opt    (ServerOpt) to update the global model on the server
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - test_freq:    (int)       how often to test UA
        - bn_setting:   (int)       private: 0=ybus, 1=yb, 2=us, 3=none
        - noisy_idxs:   (iterable)  indexes of noisy clients (ignore their UA)
        
    Returns:
        Tuple containing (train_errs, train_accs, test_errs, test_accs) as 
        Numpy arrays of length T. If test_freq > 1, non-tested rounds will 
        contain 0's.
    """
    W = len(data_feeders)

    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)

    round_model = NumpyModel(model.get_params())
    round_grads = NumpyModel(model.get_params())

    # contains private BN vals (if bn_setting != 3)
    user_bn_model_vals = [model.get_bn_vals(bn_setting) for w in range(W)]
    user_local_gate_vals = [model.get_local_gate_vals() for w in range(W)]

    for t in progressbar(range(T)):
        round_grads = round_grads.zeros_like()  # round psuedogradient

        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights /= np.sum(weights)

        round_n_test_users = 0
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model, update local model with private BN params
            model.set_params(round_model)
            model.set_bn_vals(user_bn_model_vals[user_idx], bn_setting)
            model.set_local_gate_vals(user_local_gate_vals[user_idx])
            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(test_data[0][user_idx],
                                      test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1

            # perform local SGD
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                loss, acc = model.train_step(x, y)
                train_errs[t] += loss
                train_accs[t] += acc

            # upload local model to server, store private BN params
            round_grads = round_grads + ((round_model - model.get_params()) * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(bn_setting)
            user_local_gate_vals[user_idx] = model.get_local_gate_vals()
        # update global model using psuedogradient
        round_model = server_opt.apply_gradients(round_model, round_grads)

        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users

    train_errs /= M * K
    train_accs /= M * K

    return train_errs, train_accs, test_errs, test_accs


def run_fedavg(data_feeders, test_data, model, client_opt,
               T, M, K, B, test_freq=1, bn_setting=0, noisy_idxs=[]):
    """
    Run Federated Averaging (FedAvg) algorithm from 'Communication-efficient
    learning of deep networks from decentralized data', McMahan et al., AISTATS 
    2021. In this implementation, the parameters of the client optimisers are 
    also averaged (gives FedAvg-Adam when client_opt is ClientAdam). Runs T 
    rounds of FedAvg, and returns the training and test results.
    
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - client_opt:   (ClientOpt) distributed client optimiser
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - test_freq:    (int)       how often to test UA
        - bn_setting:   (int)       private: 0=ybus, 1=yb, 2=us, 3=none
        - noisy_idxs:   (iterable)  indexes of noisy clients (ignore their UA)
        
    Returns:
        Tuple containing (train_errs, train_accs, test_errs, test_accs) as 
        Numpy arrays of length T. If test_freq > 1, non-tested rounds will 
        contain 0's.
    """

    W = len(data_feeders)

    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)

    # contains private model and optimiser BN vals (if bn_setting != 3)
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for w in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]
    user_local_gate_vals = [model.get_local_gate_vals() for w in range(W)]

    # global model/optimiser updated at the end of each round
    round_model = model.get_params()
    round_optim = client_opt.get_params()

    # stores accumulated client models/optimisers each round
    round_agg = model.get_params()
    round_opt_agg = client_opt.get_params()

    for t in progressbar(range(T)):

        round_agg = round_agg.zeros_like()
        round_opt_agg = round_opt_agg.zeros_like()

        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights /= np.sum(weights)

        round_n_test_users = 0

        for (w, user_idx) in zip(weights, user_idxs):

            # download global model/optim, update with private BN params
            model.set_params(round_model)
            client_opt.set_params(round_optim)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(user_bn_optim_vals[user_idx],
                                     model, setting=bn_setting)
            model.set_local_gate_vals(user_local_gate_vals[user_idx])

            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(test_data[0][user_idx],
                                      test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1

            # perform local SGD
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                err, acc = model.train_step(x, y)
                train_errs[t] += err
                train_accs[t] += acc

            # upload local model/optim to server, store private BN params
            round_agg = round_agg + (model.get_params() * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(model,
                                                                    setting=bn_setting)
            user_local_gate_vals[user_idx] = model.get_local_gate_vals()
        # new global model is weighted sum of client models
        round_model = round_agg.copy()
        round_optim = round_opt_agg.copy()

        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users

    train_errs /= M * K
    train_accs /= M * K

    return train_errs, train_accs, test_errs, test_accs


def run_per_fedavg(data_feeders, test_data, model, beta, T, M, K, B,
                   test_freq=1, noisy_idxs=[]):
    """
    Run Personalized-FedAvg (Per-FedAvg) algorithm from 'Personalized Federated
    Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning 
    Approach', Fallah  et al., NeurIPS 2020. Runs T rounds of Per-FedAvg, and 
    returns the test results. Note we are usign the first-order approximation 
    variant (i) described in Section 5 of Per-FedAvg paper.
    
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - beta:         (float)     parameter of Per-FedAvg algorithm
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - test_freq:    (int)       how often to test UA
        - noisy_idxs:   (iterable)  indexes of noisy clients (ignore their UA)
        
    Returns:
        Tuple containing (test_errs, test_accs) as Numpy arrays of length T. If 
        test_freq > 1, non-tested rounds will contain 0's.
    """
    W = len(data_feeders)

    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)

    # global model updated at the end of each round, and round model accumulator 
    round_model = model.get_params()
    round_agg = model.get_params()

    for t in progressbar(range(T), redirect_stdout=True):
        round_agg = round_agg.zeros_like()

        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights /= np.sum(weights)

        round_n_test_users = 0

        for (w, user_idx) in zip(weights, user_idxs):
            # download global model
            model.set_params(round_model)

            # personalise global model and test, if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                x, y = data_feeders[user_idx].next_batch(B)
                model.train_step(x, y)
                err, acc = model.test(test_data[0][user_idx],
                                      test_data[1][user_idx],
                                      128)
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1
                model.set_params(round_model)

            # perform k steps of local training, as per Algorithm 1 of paper
            for k in range(K):
                start_model = model.get_params()

                x, y = data_feeders[user_idx].next_batch(B)
                loss, acc = model.train_step(x, y)

                x, y = data_feeders[user_idx].next_batch(B)
                logits = model.forward(x)
                loss = model.loss_fn(logits, y)
                model.optim.zero_grad()
                loss.backward()

                model.set_params(start_model)
                model.optim.step(beta=beta)

            # add to round gradients
            round_agg = round_agg + (model.get_params() * w)

        # new global model is weighted sum of client models
        round_model = round_agg.copy()

        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users

    return test_errs, test_accs


def run_pFedMe(data_feeders, test_data, model, T, M, K, B, R, lamda, eta,
               beta, test_freq=1, noisy_idxs=[]):
    """
    Run pFedMe algorithm from 'Personalized Federated Learning with Moreau 
    Envelopes', Dinh et al., NeurIPS 2020. Runs T rounds of pFedMe, and returns 
    the test results. Note that, to make the algorithm comparison fair, we do 
    not activate all clients as per Algorithm 1 of the pFedMe paper, only 
    sending back the gradients of the sampled clients. Instead, we only activate 
    the sampled clients each round, brining pFedMe in line with the comparisons.
    
    Args:
        - data_feeders: (list)      of PyTorchDataFeeders
        - test_data:    (tuple)     of (x, y) testing data as torch.tensors
        - model:        (FLModel)   to run SGD on 
        - beta:         (float)     parameter of Per-FedAvg algorithm
        - T:            (int)       number of communication rounds
        - M:            (int)       number of clients sampled per round
        - K:            (int)       number of local training steps
        - B:            (int)       client batch size
        - R:            (int)       parameter R of pFedMe
        - lamda:        (float)     parameter lambda of pFedMe
        - eta:          (float)     learning rate of pFedMe
        - test_freq:    (int)       how often to test UA
        - noisy_idxs:   (iterable)  indexes of noisy clients (ignore their UA)
        
    Returns:
        Tuple containing (test_errs, test_accs) as Numpy arrays of length T. If 
        test_freq > 1, non-tested rounds will contain 0's.
    """
    W = len(data_feeders)

    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)

    # global model updated at the end of each round, and round model accumulator 
    round_model = model.get_params()
    round_agg = model.get_params()

    # client personalised models
    user_models = [round_model.copy() for w in range(W)]  # 循环创建

    for t in progressbar(range(T), redirect_stdout=True):
        round_agg = round_agg.zeros_like()

        # select round clients and compute their weights for later sum
        user_idxs = np.random.choice(W, M, replace=False)
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights /= np.sum(weights)

        round_n_test_users = 0

        for (w, user_idx) in zip(weights, user_idxs):

            # test local model if not a noisy client
            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                model.set_params(user_models[user_idx])
                err, acc = model.test(test_data[0][user_idx],
                                      test_data[1][user_idx],
                                      128)
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1

            # download global model
            model.set_params(round_model)

            # perform k steps of local training
            for r in range(R):
                x, y = data_feeders[user_idx].next_batch(B)
                omega = user_models[user_idx]
                for k in range(K):
                    model.optim.zero_grad()
                    logits = model.forward(x)
                    loss = model.loss_fn(logits, y)
                    loss.backward()
                    model.optim.step(omega)

                theta = model.get_params()

                user_models[user_idx] = omega - (lamda * eta * (omega - theta))

            round_agg = round_agg + (user_models[user_idx] * w)

        # new global model is weighted sum of client models
        round_model = (1 - beta) * round_model + beta * round_agg

        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users

    return test_errs, test_accs


def run_fedavg_adaptive(data_feeders, test_data, model, client_opt,
                        T, M, K, B, test_freq=1, bn_setting=0, noisy_idxs=[],
                        theta=0.001, lambda_=0.1, temp=1.0):
    """
    Run adaptive weighted FedAvg algorithm. Incorporates client model performance,
    data distribution difference, and convergence threshold into the aggregation.

    Additional Args:
        - theta:    (float) convergence threshold for model performance
        - lambda_:  (float) decay coefficient for data distribution difference weight
        - temp:     (float) temperature coefficient for model performance weight
    """

    W = len(data_feeders)

    train_errs, train_accs, test_errs, test_accs = init_stats_arrays(T)

    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for w in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]
    user_local_gate_vals = [model.get_local_gate_vals() for w in range(W)]

    round_model = model.get_params()
    round_optim = client_opt.get_params()

    round_agg = model.get_params()
    round_opt_agg = client_opt.get_params()

    # sample representative dataset from clients
    rep_data = sample_representative_data(data_feeders)

    for t in progressbar(range(T)):

        round_agg = round_agg.zeros_like()
        round_opt_agg = round_opt_agg.zeros_like()

        user_idxs = np.random.choice(W, M, replace=False)
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights /= np.sum(weights)

        round_n_test_users = 0
        perf_weights = np.zeros(M)
        dist_weights = np.zeros(M)
        combined_weights = np.zeros(M)

        for i, (w, user_idx) in enumerate(zip(weights, user_idxs)):

            model.set_params(round_model)
            client_opt.set_params(round_optim)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(user_bn_optim_vals[user_idx],
                                     model, setting=bn_setting)
            model.set_local_gate_vals(user_local_gate_vals[user_idx])

            if (t % test_freq == 0) and (user_idx not in noisy_idxs):
                err, acc = model.test(test_data[0][user_idx],
                                      test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
                round_n_test_users += 1

                perf_weights[i] = acc
                dist_weights[i] = calc_dist_diff(model, rep_data)

            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                err, acc = model.train_step(x, y)
                train_errs[t] += err
                train_accs[t] += acc

            round_agg = round_agg + (model.get_params() * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(model,
                                                                    setting=bn_setting)
            user_local_gate_vals[user_idx] = model.get_local_gate_vals()

        if t % test_freq == 0:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users

            perf_weights = softmax(perf_weights / temp)
            dist_weights = np.exp(-lambda_ * dist_weights)
            combined_weights = perf_weights * dist_weights
            combined_weights /= np.sum(combined_weights)  # normalize combined_weights

            round_agg *= 0
            round_opt_agg *= 0

            for i, user_idx in enumerate(user_idxs):
                if perf_weights[i] >= theta:
                    round_agg = round_agg + (model.get_params() * combined_weights[i])
                    round_opt_agg = round_opt_agg + (client_opt.get_params() * combined_weights[i])

            round_agg /= np.sum(combined_weights)
            round_opt_agg /= np.sum(combined_weights)

        round_model = round_agg.copy()
        round_optim = round_opt_agg.copy()

    train_errs /= M * K
    train_accs /= M * K

    return train_errs, train_accs, test_errs, test_accs


def sample_representative_data(data_feeders, n_samples=100):
    rep_data = []
    for feeder in data_feeders:
        x, y = feeder.next_batch(n_samples // len(data_feeders))
        rep_data.append((x.float(), y))
    return rep_data


def calc_dist_diff(model, rep_data):
    """Calculate data distribution difference"""
    diffs = []
    for x, y in rep_data:
        logits = model(x)
        probs = F.softmax(logits, dim=-1)

        # 将标签张量转换为 one-hot 编码的概率分布
        y_one_hot = F.one_hot(y, num_classes=probs.size(-1)).float()

        diffs.append(kl_div(probs, y_one_hot, reduction='batchmean'))
    return torch.mean(torch.stack(diffs))


def softmax(x, axis=-1):
    """Compute softmax values"""
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)