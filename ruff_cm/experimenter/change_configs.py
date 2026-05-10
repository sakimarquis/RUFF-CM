import os

from ruamel import yaml
from ruamel.yaml.scalarstring import SingleQuotedScalarString as sq


def change_config(path, **kwarg):
    """Apply one YAML key update to every config in a directory."""
    files = os.listdir(path)
    for file in files:
        file_path = path + file

        with open(file_path, "r", encoding='utf-8') as file:
            param = yaml.round_trip_load(file, preserve_quotes=True)

        key = kwarg['key']
        val = kwarg['val']
        val = sq(val) if type(val) is str else val
        param[key] = val

        with open(file_path, 'w', encoding="utf-8") as file:
            yaml.dump(param, file, Dumper=yaml.RoundTripDumper)


if __name__ == '__main__':
    path = '../temp/'
    change_config(path, key="DEVICE", val="cpu")
