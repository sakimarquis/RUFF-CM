from __future__ import annotations

from ruff_cm.experimenter.create_training_configs import (
    _vary_config_combinatorial,
    _vary_config_control,
    _vary_config_sequential,
    autoname,
    vary_config,
)


def base_config() -> dict:
    return {"BETA_R": 0.0, "REG": True, "ACTIVATION": "Tanh", "OPTIM_PARAMS": {"lr": 0.001, "weight_decay": 0.0}}


def test_combinatorial_count_and_values():
    configs, diffs = _vary_config_combinatorial(base_config(), {"BETA_R": [0.0001, 0.001], "REG": [True, False]})
    assert len(configs) == 4
    assert diffs == [
        {"BETA_R": 0.0001, "REG": True},
        {"BETA_R": 0.0001, "REG": False},
        {"BETA_R": 0.001, "REG": True},
        {"BETA_R": 0.001, "REG": False},
    ]


def test_sequential_count_and_values():
    configs, diffs = _vary_config_sequential(base_config(), {"BETA_R": [0.0001, 0.001], "REG": [True, False]})
    assert len(configs) == 2
    assert diffs == [{"BETA_R": 0.0001, "REG": True}, {"BETA_R": 0.001, "REG": False}]


def test_control_count_and_values():
    configs, diffs = _vary_config_control(base_config(), {"BETA_R": [0.0001, 0.001], "REG": [True, False]})
    assert len(configs) == 4
    assert diffs == [{"BETA_R": 0.0001}, {"BETA_R": 0.001}, {"REG": True}, {"REG": False}]


def test_vary_config_returns_names_not_diffs():
    configs, names = vary_config(base_config(), {"BETA_R": [0.001], "REG": [False]}, mode="combinatorial")
    assert configs[0]["BETA_R"] == 0.001
    assert names == ["BETAR-0.001_REG-False"]


def test_autoname_respects_name_keys():
    names = autoname([base_config()], [{"BETA_R": 0.001, "REG": False}], name_keys=["REG"])
    assert names == ["REG-False"]
