import logging
from abc import ABCMeta, abstractmethod
from torch.utils.tensorboard import SummaryWriter
import wandb
# import neptune.new as neptune

RECORD_INTERVAL = 100


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
        self.record_points = [0, 10, 100, 1000, 5000] + [10000 * i for i in range(1, 6)] + \
                             [100000 * i for i in range(1, 11)]

    def log_metrics(self, metrics, name, i_iter):
        if i_iter % self.record_interval == 0:
            self.logger.add_scalar(name, metrics, i_iter)

    def log_hparams(self, hparam_dict, metric_dict):
        self.logger.add_hparams(hparam_dict, metric_dict)

    def log_weights(self, model, i_iter):
        """split step to be the index of my arbitrary record points"""
        if self.logger is not None and i_iter in self.record_points:
            for name, param in model.named_parameters():
                if "weight" in name:
                    i = self.record_points.index(i_iter)
                    self.logger.add_histogram(name, param, i)

    def finish(self):
        self.logger.flush()
        self.logger.close()


class WandBLogger(ABCLogger):
    def __init__(self, project_name, expt_name, config, logger_info=None):
        if logger_info is not None:
            # Reload existing logger
            run_id = logger_info['id']
            self.logger = wandb.init(project=project_name, resume="allow", id=run_id)
            self.is_new = False
        else:
            self.logger = wandb.init(project=project_name, name=expt_name, config=config)
            self.is_new = True

    def log_metrics(self, metrics, name, i_iter):
        self.logger.log({name: metrics}, step=i_iter)

    def log_hparams(self, hparam_dict, metric_dict):
        self.logger.config.update(hparam_dict, allow_val_change=True)
        self.logger.log(metric_dict)

    def log_weights(self, model, i_iter):
        if self.logger is not None and i_iter in self.record_points:
            for name, param in model.named_parameters():
                if "weight" in name:
                    i = self.record_points.index(i_iter)
                    self.logger.log({name: wandb.Histogram(param)}, step=i)

    def finish(self):
        self.logger.finish()

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
