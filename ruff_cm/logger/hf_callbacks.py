from __future__ import annotations


class CsvLoggingCallback:
    def __new__(cls, logger):
        from transformers import TrainerCallback

        class _CsvLoggingCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    logger.log(logs, step=state.global_step)

        return _CsvLoggingCallback()
