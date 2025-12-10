#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

def setup_matplotlib_for_publication():
    rcParams['text.usetex'] = False
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 14
    rcParams['figure.figsize'] = (8, 6)
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 600
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.1
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 6
    rcParams['grid.linestyle'] = '--'
    rcParams['grid.linewidth'] = 0.5
    rcParams['grid.alpha'] = 0.8
    rcParams['axes.grid'] = False  
    rcParams['axes.axisbelow'] = True
    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 0.8
    rcParams['legend.edgecolor'] = 'k'
    rcParams['figure.facecolor'] = 'white'
    rcParams['figure.edgecolor'] = 'white'
    rcParams['axes.facecolor'] = 'white'
    rcParams['axes.edgecolor'] = 'black'
