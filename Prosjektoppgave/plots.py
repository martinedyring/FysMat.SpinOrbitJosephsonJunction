import numpy as np
import matplotlib.pyplot as plt

"""
This script contains ass plotting functions
"""

def plot_complex_function(x=None, y=None, ax=None, labels=None, **kwargs):
    #   Plot the real- and imaginary part of a function individually
    if y is None:
        raise ValueError('Y must not be None.')
    if  x is None:
        x = np.arange(y.shape[0])
    if ax is None:
        fig, ax = plt.subplots()
    if labels is None:
        labels = [None, None]

    lr = ax.plot(x, np.real(y), label=labels[0], **kwargs)
    ax.plot(np.imag(y), ls=':', c=lr[0].get_color(), label=labels[1], **kwargs)