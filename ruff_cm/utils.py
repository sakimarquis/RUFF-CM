import os
import random
import time

import numpy as np
import torch


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
