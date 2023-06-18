import os
import random
import time

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


def write_summary(loss, path, name=""):
    """write the summary of the experiment to a csv file, summary includes the loss and hyperparameters
    :param loss: loss of this run
    :param path: experiment folder where the summary will be saved, path = f"./results/{experiment}/{run}"
    :param name: name of the summary file
    """
    split = path.split("/")  # split the path by slashes
    run_name = split[-1]  # sub-folder name, which is the name of the experiment
    ex_path = f"{'/'.join(split[:-1])}"
    params_key_val = run_name.split("_")  # Split the input string by underscores
    try:
        params_val = [s.split("-")[1] for s in params_key_val]  # Extract the values after the hyphens
    except IndexError:
        params_val = []  # catch the error if there is nothing after hyphen
    if not os.path.exists(f"{ex_path}/"):
        os.makedirs(f"{ex_path}/")
    with open(f"{ex_path}{name}.csv", 'a') as file:
        file.write(f"{run_name},")
        for val in params_val:
            file.write(f"{val},")
        file.write(f"{loss},")
        file.write('\n')


def get_optimizer(params, model):
    return eval(f"optim.{params['OPTIMIZER']}")(model.parameters(), **params["OPTIM_PARAMS"])


def get_scheduler(params, optimizer):
    if params["SCHEDULER"] is None:
        return None
    elif params["SCHEDULER"] == "WarmUpLR":
        batch_size = params["BATCH_SIZE"]
        lr = params["OPTIM_PARAMS"]["lr"]
        warmup_steps = int(params["SCHED_PARAMS"]["warmup_step"] * params["N_EPOCHS"] * params["N_ITERS"])
        step_size = params["SCHED_PARAMS"]["step_size"]
        gamma = params["SCHED_PARAMS"]["gamma"]
        init_lr = lr / batch_size
        # start from lr / batch_size, then linearly increase to lr in warmup_steps, then decay by gamma after that
        lr_lambda = lambda step: init_lr + ((lr - init_lr) / warmup_steps) * step \
            if step <= warmup_steps else lr * gamma ** ((step - warmup_steps) // step_size)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return eval(f"optim.lr_scheduler.{params['SCHEDULER']}")(optimizer, **params["SCHED_PARAMS"])


def get_logger(path, debug):
    if debug:
        return Logger()
    else:
        return TensorBoardLogger(path)
