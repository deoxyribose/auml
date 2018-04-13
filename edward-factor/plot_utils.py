import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def pairplot(x):
    scattermatrix2 = sns.pairplot(pd.DataFrame(x))
    #[ax.set_ylim(-40,40) for ax in scattermatrix2.axes.flatten()]
    #[ax.set_xlim(-40,40) for ax in scattermatrix2.axes.flatten()];
    plt.show()

def doublepairplot(x,y,D):
    N = x.shape[0]
    xy = pd.DataFrame(np.c_[np.r_[x,y],np.r_[np.zeros(N),np.ones(N)]])
    names = 'abcdefghijk'[:D+1]
    xy.columns = [i for i in names]
    sns.set()
    scattermatrix = sns.pairplot(xy, hue=names[-1], diag_kind= 'kde', vars = [i for i in names[:-1]], plot_kws=dict(alpha=.2))

def plot_pairs(x):
    D = x.shape[1]
    for i in range(D-2):
        plt.scatter(*x.T[i:i+2,:],alpha=.3)
        plt.axis('equal')
        plt.show()
    plt.scatter(*x.T[(D-2):D,:],alpha=.3)
    plt.axis('equal')
    plt.show()

def plot_against_training_set(x,M):
    D = x.shape[1]
    x_train_sample = x_train[:M,:]
    for i in range(D-2):
        plt.scatter(*x_train_sample.T[i:i+2,:],alpha=.3)
        plt.scatter(*x.T[i:i+2,:],alpha=.3)
        plt.axis('equal')
        plt.show()
    plt.scatter(*x_train_sample.T[(D-2):D,:],alpha=.3)
    plt.scatter(*x.T[(D-2):D,:],alpha=.3)
    plt.axis('equal')
    plt.show()

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()