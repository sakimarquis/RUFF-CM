import os
import wandb
import numpy as np
import time
from pathlib import Path

experiment = "test_CV"
wandb.login(key=os.environ["WANDB_KEY"])
tmp = "/tmp_wandb"

cfg = {
    "hidden_size": 256,
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 100
}

loss = np.linspace(5, 0, cfg["num_epochs"])

start_time = time.time()

for different_run in range(10):
    filename = f"the_name_of_a_awesome_run_{different_run}"
    group = f"the_name_of_a_awesome_run_{different_run}"
    with wandb.init(project=experiment, name=filename, group=group, config=cfg, dir=tmp,
                    settings=wandb.Settings(_disable_stats=True, _disable_meta=True,
                                            disable_code=True, disable_git=True, silent=True,
                                            log_internal=str(Path(__file__).parent / 'wandb' / 'null'))):
        for outer_fold in range(5):
            for inner_fold in range(5):
                for epoch in range(cfg["num_epochs"]):
                    fold_info = f"outer_fold_{outer_fold}/inner_fold_{inner_fold}"
                    wandb.define_metric(f"train_loss/{fold_info}*", step_metric=f"epoch/{fold_info}")
                    wandb.define_metric(f"val_loss/{fold_info}*", step_metric=f"epoch/{fold_info}")
                    train_loss = loss[epoch] + np.random.normal(0, 0.5)
                    val_loss = train_loss + np.random.normal(0, 0.5)
                    log_dict = {
                        f"train_loss/{fold_info}": train_loss,
                        f"val_loss/{fold_info}": val_loss,
                        f"epoch/{fold_info}": epoch,
                    }
                    if epoch % 10 == 0:
                        wandb.log(log_dict)
        wandb.summary.update({"best_val_loss": np.random.normal(0, 0.5),
                              "best_test_loss": np.random.normal(0, 0.5)})

print(f"Time cost：{time.time() - start_time}s")
