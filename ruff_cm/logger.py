import os
import logging
from pathlib import Path
from abc import ABCMeta, abstractmethod
from torch.utils.tensorboard import SummaryWriter
import wandb
# import neptune.new as neptune

RECORD_INTERVAL = 100
WEIGHTS_INTERVAL = ([0, 10, 100, 1000, 5000] + [10000 * i for i in range(1, 6)] +
                    [100000 * i for i in range(1, 11)])


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
    """a dummy logger that does nothing"""
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
    """a wrapper for a python logging console handler"""
    def __init__(self, logger_name="Iter", record_interval=RECORD_INTERVAL):
        self.record_interval = record_interval
        self.name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler_format = '%(message)s'
        console_handler.setFormatter(logging.Formatter(console_handler_format))
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
                    flatten_weights = param.view(-1)
                    self.logger.debug(f"{self.name} {i_iter} - {name}: {flatten_weights}")

    def finish(self):
        """remove all handlers and close the logger"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


class TensorBoardLogger(ABCLogger):
    """A thin wrapper for SummaryWriter"""
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
        """split step to be the index of my arbitrary record points"""
        if self.logger is not None and i_iter in self.weights_record_points:
            for name, param in model.named_parameters():
                if "weight" in name:
                    i = self.weights_record_points.index(i_iter)
                    self.logger.add_histogram(name, param, i)

    def finish(self):
        self.logger.flush()
        self.logger.close()


class WandBLogger(ABCLogger):
    """The logic of WandBLogger is very different from the other loggers, since we use cross validation.
    So, each run contains all folds (inner or outer).
    If different folds are different runs, it would be hard to aggregate the results, and a cluttered dashboard.

    We have to define custom metrics, and log them at the end of each fold.
    https://docs.wandb.ai/guides/track/log/customize-logging-axes
    """
    def __init__(self, config, name, record_interval):
        self.record_interval = record_interval
        self.weights_record_points = WEIGHTS_INTERVAL
        self.logger_name = name
        self.fold_info = self._get_fold_info(config)


        wandb.define_metric(f"Loss/{self.fold_info}*", step_metric=f"Epoch/{self.fold_info}")
        wandb.define_metric(f"Accuracy/{self.fold_info}*", step_metric=f"Epoch/{self.fold_info}")
        wandb.define_metric(f"ValLoss/{self.fold_info}*", step_metric=f"Epoch/{self.fold_info}")
        wandb.define_metric(f"ValAccuracy/{self.fold_info}*", step_metric=f"Epoch/{self.fold_info}")
        wandb.define_metric(f"GradNorm/{self.fold_info}*", step_metric=f"Epoch/{self.fold_info}")
        wandb.define_metric(f"i_action_loss/{self.fold_info}*", step_metric=f"Iter/{self.fold_info}")

    def _get_fold_info(self, config):
        self.fold_info = ""
        fold = config.get("FOLD")
        outer_fold = config.get("OUTER_FOLD")
        sub = config.get("SUB")
        trainer = config.get("TRAINER")
        if sub is not None and "sub" in trainer.lower():
            self.fold_info += f"sub{sub}/"
        if outer_fold is not None:
            self.fold_info += f"outer_fold{outer_fold}/"
        if fold is not None:
            self.fold_info += f"fold{fold}"

    def log_metrics(self, metrics, name, i_iter):
        """
        :param metrics: a value
        :param name: the name of the metric
        :param i_iter: the step of the metric
        """
        if i_iter % self.record_interval == 0:
            wandb.log({f"{name}/{self.fold_info}": metrics,
                       f"{self.logger_name}/{self.fold_info}": i_iter})

    def log_hparams(self, hparam_dict, metric_dict):
        """we already log hparams in the beginning of each run, so we don't need to log it again"""
        fold_metric_dict = {f"{key}/{self.fold_info}": value for key, value in metric_dict.items()}
        wandb.summary.update(fold_metric_dict)

    def log_weights(self, model, i_iter):
        if i_iter in self.weights_record_points:
            for name, param in model.named_parameters():
                if "weight" in name:
                    i = self.weights_record_points.index(i_iter)
                    wandb.log({f"{name}/{self.fold_info}": wandb.Histogram(param.detach().cpu().numpy().flatten()),
                               f"{self.logger_name}/{self.fold_info}": i})

    def finish(self):
        """we don't need to do anything"""
        pass


def wandb_run_trainer(trainer, config, experiment, filename, silent=True):
    """https://github.com/wandb/wandb/issues/4223#issuecomment-1236304565"""
    wandb.login(key=os.environ["WANDB_KEY"])
    log_dir = config.get("LOG_CACHE_DIR", os.getcwd())
    with wandb.init(project=experiment, name=filename.replace('.yml', ''), group=filename.replace('.yml', ''),
                    config=config, dir=log_dir, settings=wandb.Settings(
                _disable_stats=True, _disable_meta=True, disable_code=True, disable_git=True, silent=silent,
                # log_internal=str(Path(__file__).parent / 'wandb' / 'null')),
                log_internal=str(Path(os.getcwd()) / 'wandb' / 'null'))):
        trainer.run()

# class NeptuneLogger(ABCLogger):
#     # A thin wrapper for Neptune's logger
#     def __init__(self, logger_info, neptune_project, expt_name,
#                  params, tags):
#         if logger_info is not None:
#             # Reload existing logger
#             expt_id = logger_info['id']
#             self.logger = neptune.init(project=neptune_project, run=expt_id)
#             self.is_new = False
#         else:
#             self.logger = neptune.init(project=neptune_project, name=expt_name)
#             self.logger['parameters'] = params
#             self.logger['sys/tags'].add(tags)
#             self.info = self.logger.fetch()['sys']
#             self.is_new = True
#
#     def log_metrics(self, metrics, name, epoch, epoch_end=False,
#                     iteration=None, anneal_param=None):
#         for key, val in metrics.items():
#             self.logger[f'{name}_{key}'].log(step=epoch, value=val)
#         if epoch_end:
#             self.logger['iteration'].log(step=epoch, value=iteration)
#             self.logger['anneal_param'].log(step=epoch, value=anneal_param)
#
#     def log_sample_output(self, fig, epoch):
#         assign_key = f'model_outputs/epoch{epoch}'
#         self.logger[assign_key].log(fig)
