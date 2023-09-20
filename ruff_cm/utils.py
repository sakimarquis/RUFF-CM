import os
import random
import time
import hashlib
import yaml
import psutil

import numpy as np
import torch
import torch.optim as optim

from .logger import Logger, TensorBoardLogger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def timer(func):
    def wrapper(*args, **kw):
        t1 = time.time()
        func(*args, **kw)
        t2 = time.time()
        cost_time = t2 - t1
        print("Time cost：{}s".format(cost_time))

    return wrapper


def write_summary(path, loss, accuracy=None, suffix=""):
    """write the summary of the experiment to a csv file, summary includes the loss and hyperparameters
    :param path: experiment folder where the summary will be saved, path = f"./results/{experiment}/{run}"
    :param loss: float or a list of float, loss of this run
    :param accuracy: float or a list of float, accuracy of this run
    :param suffix: suffix name of the summary file
    """
    split = path.split("/")  # split the path by slashes
    run_name = split[-1]  # sub-folder name, which is the name of the experiment
    ex_path = f"{'/'.join(split[:-1])}"
    params_key_val = run_name.split("_")  # Split the input string by underscores
    params_val = _get_params_val(params_key_val)

    if not os.path.exists(f"{ex_path}/"):
        os.makedirs(f"{ex_path}/")

    with open(f"{ex_path}{suffix}.csv", 'a') as file:
        file.write(f"{run_name},")
        _write_val(params_val, file)
        _write_val(loss, file)
        if accuracy is not None:
            _write_val(accuracy, file)
        # file.write(f"{loss},")
        file.write('\n')


def _get_params_val(params_key_val):
    """get the params values from the config file, if the config file is not formatted correctly, return empty list
    e.g. params_key_val = ["lr-0.001", "batch_size-32", "optimizer-Adam", "scheduler-WarmUpLR"]
    """
    try:
        params_val = [s.split("-")[1] for s in params_key_val]  # Extract the values after the hyphens
    except IndexError:
        params_val = []  # catch the error if there is nothing after hyphen
    return params_val


def _write_val(val, file):
    if isinstance(val, list):  # Check if 'loss' is a list of losses
        for v in val:
            file.write(f"{v},")
    else:
        file.write(f"{val},")


def get_optimizer(params, model):
    freeze_layers = params.get("FREEZE_LAYERS", None)
    if isinstance(freeze_layers, list) and len(freeze_layers) > 0:
        trainable_parameters = [name for name, param in model.named_parameters() if param.requires_grad]
        print(f"Trainable parameters: {trainable_parameters}")
    return eval(f"optim.{params['OPTIMIZER']}")(filter(lambda p: p.requires_grad, model.parameters()), **params["OPTIM_PARAMS"])


def get_scheduler(params, optimizer):
    if params["SCHEDULER"] is None:
        return None
    elif params["SCHEDULER"] == "WarmUpLR":
        batch_size = params["BATCH_SIZE"]
        lr = params["OPTIM_PARAMS"]["lr"]
        warmup_steps = int(params["SCHED_PARAMS"]["warmup_step"] * params["N_EPOCHS"] * params.get("N_ITERS", 1))
        step_size = params["SCHED_PARAMS"]["step_size"]
        gamma = params["SCHED_PARAMS"]["gamma"]
        init_lr = lr / batch_size
        # start from lr / batch_size, then linearly increase to lr in warmup_steps, then decay by gamma after that
        lr_lambda = lambda step: init_lr + ((lr - init_lr) / warmup_steps) * step \
            if step <= warmup_steps else lr * gamma ** ((step - warmup_steps) // step_size)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return eval(f"optim.lr_scheduler.{params['SCHEDULER']}")(optimizer, **params["SCHED_PARAMS"])


def get_logger(path=None, debug=False, name="Epoch", record_interval=100):
    if debug:
        return Logger(name, record_interval)
    else:
        return TensorBoardLogger(path, record_interval)


def dump_yaml(dict_to_dump, path, name):
    with open(f'{path}/{name}.yml', 'w', encoding="utf-8") as file:
        yaml.dump(dict_to_dump, file, Dumper=yaml.Dumper, default_flow_style=False, sort_keys=False)


def hash_string(input_string, uid: str, algorithm='md5', truncate: int = 12):
    """hash the subject id and return a unique id for the subject"""
    unique_string = input_string + uid  # append a unique identifier
    hash_object = hashlib.new(algorithm)
    hash_object.update(unique_string.encode('utf-8'))
    return hash_object.hexdigest()[:truncate]


def print_ram_usage(idx, use_cuda=False):
    ram = psutil.virtual_memory()
    ram_percent = (ram.total - ram.available) / ram.total * 100
    swap_percent = psutil.swap_memory().percent
    print(f"RAM usage: {ram_percent:.2f}%, swap usage: {swap_percent:.2f}%, idx: {idx}")
    if torch.cuda.is_available() and use_cuda:
        vram_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        print(f"GPU memory usage: {vram_percent * 100:.2f}%")
