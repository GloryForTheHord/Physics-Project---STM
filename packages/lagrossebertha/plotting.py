import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot(x :np.ndarray, *args :tuple, **kwargs):
    
    needLegend = False
    
    for arg in args:
        if(arg[2] != ''):
            needLegend = True
        plt.plot(x, arg[0], arg[1], label=arg[2])  
        
    if needLegend:
        plt.legend(loc="best")
        
    plt.grid(linewidth = 0.5, linestyle = "--")
        
    for key, value in kwargs.items():
        if key == "title":
            plt.title(r'{}'.format(value), fontsize=13, pad=10)
        if key == "suptitle":
            plt.suptitle(r'{}'.format(value), fontsize=13)
        if key == "xlabel":
            plt.xlabel(value, fontsize=10)
        if key == "ylabel":
            plt.ylabel(value, fontsize=10)
        if key == "loc":
            plt.legend(loc=value)
        if key == "grid":
            plt.grid(**value)