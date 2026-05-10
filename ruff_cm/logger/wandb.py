from __future__ import annotations

import json
import logging
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from ruff_cm.experimenter.io import from_portable_relpath, portable_relpath

RECORD_INTERVAL = 100
WEIGHTS_INTERVAL = [0, 10, 100, 1000, 5000] + [10000 * i for i in range(1, 6)] + [100000 * i for i in range(1, 11)]


def _import_wandb():
    import wandb

    return wandb


class WandbLogger:
    def __init__(self, run=None, *, base_dir: Path | str | None = None):
        self.run = run
        self.base_dir = Path(base_dir) if base_dir is not None else None

    @classmethod
    def start(cls, *, project: str, run_name: str, config: dict, base_dir: Path | str | None = None) -> "WandbLogger":
        run = _import_wandb().init(project=project, id=run_name, name=run_name, config=config, resume="allow")
        return cls(run, base_dir=base_dir)

    @classmethod
    def resume(cls, *, project: str, run_name: str, base_dir: Path | str | None = None) -> "WandbLogger":
        run = _import_wandb().init(project=project, id=run_name, resume="must")
        return cls(run, base_dir=base_dir)

    @property
    def config(self) -> dict:
        return dict(self.run.config)

    def log(self, metrics: dict, *, step: int | None = None) -> None:
        self.run.log(metrics, step=step)

    def set_summary(self, metrics: dict) -> None:
        for key, value in metrics.items():
            self.run.summary[key] = value

    def record_ckpt(self, ckpt_path: Path, *, extras: dict | None = None) -> None:
        run_dir = self.base_dir or Path(self.run.dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {"ckpt_rel": portable_relpath(Path(ckpt_path), run_dir)}
        if extras:
            payload.update(extras)
        (run_dir / "latest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        for key, value in payload.items():
            self.run.summary[key] = value

    def get_ckpt(self) -> Path | None:
        run_dir = self.base_dir or Path(self.run.dir)
        latest_path = run_dir / "latest.json"
        if not latest_path.exists():
            return None
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        return from_portable_relpath(payload["ckpt_rel"], run_dir)

    def hf_report_to(self) -> list[str]:
        return ["wandb"]

    def hf_callbacks(self) -> list:
        from transformers.integrations import WandbCallback

        return [WandbCallback()]

    def finish(self) -> None:
        self.run.finish()


class ABCLogger(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def log_metrics(self, metrics, name, i_iter):
        pass

    @abstractmethod
    def log_hparams(self, hparam_dict, metric_dict):
        pass

    @abstractmethod
    def log_weights(self, model, i_iter):
        pass

    @abstractmethod
    def finish(self):
        pass


class DummyLogger(ABCLogger):
    def __init__(self, *args):
        pass

    def log_metrics(self, metrics, name, i_iter):
        pass

    def log_hparams(self, hparam_dict, metric_dict):
        pass

    def log_weights(self, model, i_iter):
        pass

    def finish(self):
        pass


class Logger(ABCLogger):
    """Console scalar logger for the `ruff_cm.logger.Logger` import."""

    def __init__(self, logger_name="Iter", record_interval=RECORD_INTERVAL):
        self.record_interval = record_interval
        self.name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)

    def log_metrics(self, metrics, name, i_iter):
        if i_iter % self.record_interval == 0:
            self.logger.info(f"{self.name} {i_iter} - {name:<12s}: {metrics:.4f}")

    def log_hparams(self, hparam_dict, metric_dict):
        self.logger.debug(f"Hyper-parameters: {hparam_dict}")

    def log_weights(self, model, i_iter):
        if i_iter % self.record_interval == 0:
            for name, param in model.named_parameters():
                if "weight" in name:
                    self.logger.debug(f"{self.name} {i_iter} - {name}: {param.view(-1)}")

    def finish(self):
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


class TensorBoardLogger(ABCLogger):
    def __init__(self, path, record_interval=RECORD_INTERVAL):
        self.record_interval = record_interval
        self.logger = SummaryWriter(path)
        self.weights_record_points = WEIGHTS_INTERVAL

    def log_metrics(self, metrics, name, i_iter):
        if i_iter % self.record_interval == 0:
            self.logger.add_scalar(name, metrics, i_iter)

    def log_hparams(self, hparam_dict, metric_dict):
        self.logger.add_hparams(hparam_dict, metric_dict)

    def log_weights(self, model, i_iter):
        if self.logger is not None and i_iter in self.weights_record_points:
            for name, param in model.named_parameters():
                if "weight" in name:
                    self.logger.add_histogram(name, param, self.weights_record_points.index(i_iter))

    def finish(self):
        self.logger.flush()
        self.logger.close()


class WandBLogger(ABCLogger):
    """Cross-validation WandB logger with lazy wandb import."""

    def __init__(self, config, name, record_interval):
        self.record_interval = record_interval
        self.weights_record_points = WEIGHTS_INTERVAL
        self.logger_name = name
        self.fold_info = self._get_fold_info(config)
        wandb = _import_wandb()
        wandb.define_metric(f"Loss/{self.fold_info}*", step_metric=f"Epoch/{self.fold_info}")
        wandb.define_metric(f"Accuracy/{self.fold_info}*", step_metric=f"Epoch/{self.fold_info}")
        wandb.define_metric(f"ValLoss/{self.fold_info}*", step_metric=f"Epoch/{self.fold_info}")
        wandb.define_metric(f"ValAccuracy/{self.fold_info}*", step_metric=f"Epoch/{self.fold_info}")
        wandb.define_metric(f"GradNorm/{self.fold_info}*", step_metric=f"Epoch/{self.fold_info}")
        wandb.define_metric(f"i_action_loss/{self.fold_info}*", step_metric=f"Iter/{self.fold_info}")

    @staticmethod
    def _get_fold_info(config):
        fold = config.get("FOLD")
        outer_fold = config.get("OUTER_FOLD")
        sub = config.get("SUB")
        trainer = config.get("TRAINER")
        fold_info = ""
        if sub is not None and "sub" in trainer.lower():
            fold_info += f"sub{sub}/"
        if outer_fold is not None:
            fold_info += f"outer_fold{outer_fold}/"
        if fold is not None:
            fold_info += f"fold{fold}"
        return fold_info

    def log_metrics(self, metrics, name, i_iter):
        if i_iter % self.record_interval == 0:
            _import_wandb().log({f"{name}/{self.fold_info}": metrics, f"{self.logger_name}/{self.fold_info}": i_iter})

    def log_hparams(self, hparam_dict, metric_dict):
        _import_wandb().summary.update({f"{key}/{self.fold_info}": value for key, value in metric_dict.items()})

    def log_weights(self, model, i_iter):
        if i_iter in self.weights_record_points:
            wandb = _import_wandb()
            for name, param in model.named_parameters():
                if "weight" in name:
                    step = self.weights_record_points.index(i_iter)
                    values = param.detach().cpu().numpy().flatten()
                    weight_key = f"{name}/{self.fold_info}"
                    step_key = f"{self.logger_name}/{self.fold_info}"
                    wandb.log({weight_key: wandb.Histogram(values), step_key: step})

    def finish(self):
        pass


def wandb_run_trainer(trainer, config, project_name, config_file, silent=True, default_log_dir=False):
    """Run a trainer inside a WandB run."""
    trainer.logger_type = "wandb"
    if default_log_dir:
        log_dir = None
        log_internal = None
    else:
        log_dir = config.get("LOG_CACHE_DIR", os.getcwd())
        log_internal = str(Path(os.getcwd()) / "wandb" / "null")
    wandb = _import_wandb()
    wandb.login(key=os.environ["WANDB_KEY"])
    with wandb.init(project=project_name, name=config_file.replace(".yml", ""), group=config_file.replace(".yml", ""),
                    config=config, dir=log_dir, settings=wandb.Settings(_disable_stats=True, _disable_meta=True,
                    disable_code=True, disable_git=True, silent=silent, log_internal=log_internal)):
        metrics = trainer.run()
        wandb.summary.update(metrics)
    return metrics.get("test_nll_mean", metrics.get("val_nll_mean", -1))
