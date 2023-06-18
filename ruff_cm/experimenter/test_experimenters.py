from experimenters.create_training_configs import *
from experimenters.create_training_configs import _vary_config_combinatorial, _vary_config_sequential, \
    _vary_config_control


def test_combinatorial(param):
    ex_params = {
        "BETA_R": [0.0001, 0.001, 0.005],
        "REG": [True, False],
        "ACTIVATION": ["Tanh", "ReLU"]
    }
    configs, config_diff1 = _vary_config_combinatorial(param, ex_params)
    save_config(configs[0], path, "test2.yml")
    print(config_diff1)


def test_sequential(param):
    ex_params = {
        "BETA_R": [0.0001, 0.001],
        "REG": [True, False],
        "ACTIVATION": ["Tanh", "ReLU"]
    }
    configs, config_diff2 = _vary_config_sequential(param, ex_params)
    save_config(configs[0], path, "test3.yml")
    print(config_diff2)


def test_control(param):
    ex_params = {
        "BETA_R": [0.0001, 0.001],
        "REG": [True, False],
    }
    configs, config_diff3 = _vary_config_control(param, ex_params)
    save_config(configs[0], path, "test4.yml")
    print(config_diff3)


if __name__ == '__main__':
    path = "../temp/"
    filename = "test.yml"

    param = load_config(path)
    save_config(param, path, "test1.yml")

    test_combinatorial(param)
    test_sequential(param)
    test_control(param)
