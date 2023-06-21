# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:08:53 2022

@author: saki
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


PLOT_PARAMS = {"dpi": 300,
               "bbox_inches": 'tight',
               "pad_inches": 0.1,
               "transparent": False}

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.family'] = 'arial'


def set_mpl():
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'arial'
    mpl.rcParams['savefig.dpi'] = 600
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False


def plot_start(square=True, figsize=None, ticks_pos=True):
    """unified plot params"""
    set_mpl()
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    elif square:
        fig = plt.figure(figsize=(1.5, 1.5))
    else:
        fig = plt.figure(figsize=(1.5, 0.8))
    ax = fig.add_axes((0.1,0.1,0.8,0.8))
    if ticks_pos:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    return fig, ax


def set_background_color(figure=(1.0, 1.0, 1.0, 1), axes=(1.0, 1.0, 1.0, 1), color=(1.0, 1.0, 1.0, 1)):
    """https://stackoverflow.com/a/62466259/8141593
    input: r, g, b, alpha
    """
    plt.rcParams.update({
        "figure.facecolor": figure,
        "axes.facecolor": axes,
        "savefig.facecolor": color,
    })
