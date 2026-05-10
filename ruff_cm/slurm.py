import submitit
import itertools
from typing import List, Dict, Any, Callable
from .utils import load_yaml

SAVE_DIR = "/home/u1/hdx/"


def factorize_configs(config_ranges: Dict[str, List[Any]]) -> List[Dict]:
    """Expand config ranges into one config dict per cartesian-product cell."""
    keys, values = zip(*config_ranges.items())
    return [dict(zip(keys, values)) for values in itertools.product(*values)]


def create_run_name(config: Dict) -> str:
    """Create a stable run name from scalar training config fields."""
    cfg = config.copy()
    if "OPTIM_PARAMS" in cfg.keys():
        optim_params = cfg.pop("OPTIM_PARAMS")
        cfg["LR"] = optim_params["lr"]
        cfg["WD"] = optim_params["weight_decay"]
    keys, values = zip(*cfg.items())
    keys = [k.replace("_", "") for k in keys]
    name = [f"{key}-{value}" for key, value in zip(keys, values)]
    name = "_".join(name)
    return name


def create_experiments_configs(
        base_config: str,
        config_ranges: Dict[str, List[Any]],
        experiment_name: str,
        logger_type: str = "dummy",
        max_jobs: int = 1000,
        save_dir: str = SAVE_DIR,
) -> List[Dict]:
    """Create experiment configs for submission to the Slurm cluster."""
    override_configs = factorize_configs(config_ranges)
    path = f"./configs/{base_config}"
    param = load_yaml(path)
    param["LOG_CACHE_DIR"] = f"/{save_dir}/logdir/"
    param["LOGGER"] = logger_type

    configs = []
    for config in override_configs:
        new_param = param.copy()
        new_param.update(config)
        run_name = create_run_name(config)
        new_param["RUN_NAME"] = run_name
        new_param["EX_NAME"] = experiment_name
        new_param["PATH"] = f"./results/{experiment_name}/{run_name}"
        configs.append(new_param)

    print(f"Submitting {len(configs)} jobs...")
    assert len(configs) < max_jobs, f"Too many jobs! {len(configs)}"
    return configs


def run(func: Callable, ex_configs: List[Dict], slurm_config: Dict[str, Any], save_dir: str = SAVE_DIR):
    for config in ex_configs:
        log_dir = f"/{save_dir}/temp/{config['RUN_NAME']}"
        slurm_config['name'] = config['RUN_NAME']
        executor = submitit.AutoExecutor(folder=log_dir)
        executor.update_parameters(**slurm_config)
        job = executor.submit(func, config)
        print(job.job_id)


def batch_run(func: Callable, ex_configs: List[Dict], slurm_config: Dict[str, Any],
              batch_size: int = 50, save_dir: str = SAVE_DIR):
    """Submit experiment configs to Slurm in fixed-size batches."""
    log_dir = f"/{save_dir}/temp/{slurm_config['name']}"
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(**slurm_config)
    for i in range(0, len(ex_configs), batch_size):
        jobs = executor.map_array(func, ex_configs[i:i + batch_size])
        for job in jobs:
            print(job.job_id)
