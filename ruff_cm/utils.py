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


def write_summary(path: str, metrics: Dict[str, List]):
    """write the summary of the experiment to a csv file, summary includes the loss and hyperparameters
    :param path: pattern f"./results/{experiment}/{run}", experiment folder where the summary will be saved
    :param metrics: keys are metric names, values are lists of metric values
    """
    for v in metrics.values():
        assert isinstance(v, (list, tuple, np.ndarray)), "values in metrics should be list, tuple or np.ndarray"
    components = os.path.normpath(path).split(os.sep)
    idx = components.index("results")
    ex_path = f"{'/'.join(components[:idx+2])}"
    run_name = components[idx+2] if len(components[idx:]) > 2 else "run"  # sub-folder name is the name of the run
    params_key, params_val = get_hyperparams_from_name(run_name)

    local_path = f"{ex_path}/{run_name}"
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    _write_summary_csv(run_name, params_key, params_val, metrics, f"{local_path}/perf.csv")
    _write_summary_csv(run_name, params_key, params_val, metrics, f"{ex_path}_perf.csv")


def _write_summary_csv(run_name, params_key, params_val, metrics, summary_file):
    """write the summary of the experiment to a csv file, summary includes the loss and hyperparameters"""
    file_exist = os.path.isfile(summary_file)  # codes below will create the summary_file

    with open(summary_file, 'a+', newline='') as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist
        if not file_exist:
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


def get_hyperparams_from_name(run_name):
    """get the params values from the config file, if the config file is not formatted correctly, return empty list
    
    example:
        run_name = "lr-0.001_batch_size-32_optimizer-Adam_scheduler-WarmUpLR"
        params_key = ["lr", "batch_size", "optimizer", "scheduler"]; params_val = ["0.001", "32", "Adam", "WarmUpLR"]
    """
    params_key_val = run_name.split("_")  # Split the input string by underscores
    try:
        params_key = [s.split("-")[0] for s in params_key_val]  # Extract the keys before the hyphens
        params_val = [s.split("-")[1] for s in params_key_val]  # Extract the values after the hyphens
    except IndexError:
        params_key = []  # catch the error if there is nothing before hyphen
        params_val = []  # catch the error if there is nothing after hyphen
    return params_key, params_val


def get_optimizer(params, model):
    decay_params = []
    non_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if hasattr(param, 'decay') and not param.decay:
                non_decay_params.append(param)
            else:
                decay_params.append(param)

    freeze_layers = params.get("FREEZE_LAYERS", None)
    if isinstance(freeze_layers, list) and len(freeze_layers) > 0:
        trainable_parameters = [name for name, param in model.named_parameters() if param.requires_grad]
        print(f"Trainable parameters: {trainable_parameters}")

    if decay_params and non_decay_params:
        print(f"Non-decay parameters: {len(non_decay_params)}")
        parameter_groups = [{'params': decay_params}, {'params': non_decay_params, 'weight_decay': 0.0}]
    else:
        parameter_groups = filter(lambda p: p.requires_grad, model.parameters())

    return eval(f"optim.{params['OPTIMIZER']}")(parameter_groups, **params["OPTIM_PARAMS"])


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


def get_cache_dir(path, cache_dir):
    """store the log files in the cache directory"""
    expt_dir = get_expt_name(path)
    expt_cache_dir = cache_dir + expt_dir
    return expt_cache_dir


def get_save_dir(path, save_dir):
    directory = get_expt_name(path)
    path = save_dir + directory
    return path


def dump_yaml(dict_to_dump: Dict, path: str, file_name: str):
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.preserve_quotes = True
    with open(f'{path}/{file_name}.yml', 'w', encoding="utf-8") as file:
        yaml.dump(dict_to_dump, file)


def load_yaml(file_path: str) -> Dict:
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(file_path, 'r', encoding='utf-8') as file:
        yaml_dict = yaml.load(file)
    return yaml_dict


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
