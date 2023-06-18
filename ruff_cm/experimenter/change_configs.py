# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:31:24 2022

@author: hdx
"""

import os

from ruamel import yaml
from ruamel.yaml.scalarstring import SingleQuotedScalarString as sq


def change_config(path, **kwarg):
    """if there is no param add it, if there is not, change param specification"""
    files = os.listdir(path)
    for file in files:
        file_path = path + file

        with open(file_path, "r", encoding='utf-8') as file:
            param = yaml.round_trip_load(file, preserve_quotes=True)

        key = kwarg['key']
        val = kwarg['val']
        # for string, should use sq("string") to avoid the problem of yaml
        val = sq(val) if type(val) is str else val
        param[key] = val  # change values

        with open(file_path, 'w', encoding="utf-8") as file:
            yaml.dump(param, file, Dumper=yaml.RoundTripDumper)


if __name__ == '__main__':
    path = '../temp/'
    change_config(path, key="DEVICE", val="cpu")
