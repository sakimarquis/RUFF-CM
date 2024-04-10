import os
import re
import random
import time
import hashlib
import csv
from typing import List, Dict, Callable
from ruamel.yaml import YAML

import numpy as np
import torch
import torch.optim as optim
# import psutil

from .logger import Logger, TensorBoardLogger, DummyLogger, WandBLogger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def timer(func: Callable):
    def wrapper(*args, **kw):
        t1 = time.time()
        func(*args, **kw)
        t2 = time.time()
        cost_time = t2 - t1
        print("Time cost：{}s".format(cost_time))

    return wrapper


def write_summary(path: str, metrics: Dict[str, List], suffix: str = "_summary"):
    """write the summary of the experiment to a csv file, summary includes the loss and hyperparameters
    :param path: experiment folder where the summary will be saved, path = f"./results/{experiment}/{run}"
    :param metrics: keys are metric names, values are lists of metric values
    :param suffix: suffix name of the summary file
    """
    split = path.split("/")  # split the path by slashes
    run_name = split[-1]  # sub-folder name, which is the name of the experiment
    ex_path = f"{'/'.join(split[:-1])}"
    params_key_val = run_name.split("_")  # Split the input string by underscores
    params_key, params_val = _get_params(params_key_val)
    save_run_summary(ex_path, run_name, params_key, params_val, metrics, "", save="local")
    save_run_summary(ex_path, run_name, params_key, params_val, metrics, suffix, save="global")


def save_run_summary(ex_path, run_name, params_key, params_val, metrics, suffix, save="local"):
    """write the summary of the experiment to a csv file, summary includes the loss and hyperparameters"""
    if save == "local":  # save the run in the run folder
        if len(run_name) == 0:
            run_name = "run"  # only when no run name is specified when testing
        local_path = f"{ex_path}{suffix}/{run_name}/"
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        summary_file = f"{local_path}/run_perf.csv"
    else:
        summary_file = f"{ex_path}{suffix}.csv"
    file_exists = os.path.isfile(summary_file)

    with open(summary_file, 'a+', newline='') as file:
        writer = csv.writer(file)
        # Write the header only if the file does not exist
        if not file_exists:
            header = ['ex_name'] + params_key
            for key, value in metrics.items():
                if len(value) > 1:
                    header.extend(f"{key}_fold{i}" for i in range(len(value)))
                else:
                    header.append(key)
            writer.writerow(header)
        # write the hyperparameters and metrics
        row = [run_name] + params_val
        for value in metrics.values():
            row.extend(value)
        writer.writerow(row)


def _get_params(params_key_val):
    """get the params values from the config file, if the config file is not formatted correctly, return empty list
    e.g. params_key_val = ["lr-0.001", "batch_size-32", "optimizer-Adam", "scheduler-WarmUpLR"]
    """
    try:
        params_key = [s.split("-")[0] for s in params_key_val]  # Extract the keys before the hyphens
        params_val = [s.split("-")[1] for s in params_key_val]  # Extract the values after the hyphens
    except IndexError:
        params_key = []  # catch the error if there is nothing before hyphen
        params_val = []  # catch the error if there is nothing after hyphen
    return params_key, params_val


def get_optimizer(params, model):
    freeze_layers = params.get("FREEZE_LAYERS", None)
    if isinstance(freeze_layers, list) and len(freeze_layers) > 0:
        trainable_parameters = [name for name, param in model.named_parameters() if param.requires_grad]
        print(f"Trainable parameters: {trainable_parameters}")
    return eval(f"optim.{params['OPTIMIZER']}")(filter(lambda p: p.requires_grad, model.parameters()),
                                                **params["OPTIM_PARAMS"])


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


def get_logger(path=None, logger="debug", name="Epoch", record_interval=100, config=None):
    if logger is None or logger == "debug":
        return Logger(name, record_interval)
    elif logger == "dummy":
        return DummyLogger()
    elif logger == "tensorboard":
        log_cache_dir = config.get("LOG_CACHE_DIR")
        if log_cache_dir is not None:
            path = get_cache_dir(path, log_cache_dir)
        return TensorBoardLogger(path, record_interval)
    elif logger == "wandb":
        return WandBLogger(config, name, record_interval)
    else:
        raise ValueError(f"Logger {logger} not supported")


def get_expt_name(path):
    pattern = r"/results/(.*)"  # search for the directory after '/results/'
    match = re.search(pattern, path)

    if match:
        directory = match.group(1)
        return directory
    else:
        raise ValueError(f"cannot find experiment name from path {path}")


def get_cache_dir(path, cache_dir="/xdisk/bob/hdx/logdir/"):
    """store the log files in the cache directory"""
    expt_dir = get_expt_name(path)
    expt_cache_dir = cache_dir + expt_dir
    return expt_cache_dir


def get_save_dir(path, save_dir="/groups/bob/hdx/"):
    directory = get_expt_name(path)
    path = save_dir + directory
    return path


def dump_yaml(dict_to_dump, path, name):
    yaml = YAML()
    yaml.default_flow_style = False
    with open(f'{path}/{name}.yml', 'w', encoding="utf-8") as file:
        yaml.dump(dict_to_dump, file)


def hash_string(input_string, uid: str, algorithm='md5', truncate: int = 12):
    """hash the subject id and return a unique id for the subject"""
    unique_string = input_string + uid  # append a unique identifier
    hash_object = hashlib.new(algorithm)
    hash_object.update(unique_string.encode('utf-8'))
    return hash_object.hexdigest()[:truncate]


def calculate_bic(likelihood: float, num_params: int, sample_size: int) -> float:
    bic = -2 * np.log(likelihood) + num_params * np.log(sample_size)
    return bic


# def print_ram_usage(idx, use_cuda=False):
#     ram = psutil.virtual_memory()
#     ram_percent = (ram.total - ram.available) / ram.total * 100
#     swap_percent = psutil.swap_memory().percent
#     print(f"RAM usage: {ram_percent:.2f}%, swap usage: {swap_percent:.2f}%, idx: {idx}")
#     if torch.cuda.is_available() and use_cuda:
#         vram_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
#         print(f"GPU memory usage: {vram_percent * 100:.2f}%")
