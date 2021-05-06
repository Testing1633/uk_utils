import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import multivariate_normal


# -----------------------------------------------------------

def plot_decision_regions(X, y, classifier, x1range=None, x2range=None, x1=0, x2=1, gridpoints=100, figsize=(6, 6)):
    # Select colormaps
    # markers = ('s', 'x', 'o', '^', 'v')
    # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap1 = plt.cm.Pastel1  # ListedColormap(colors[:len(np.unique(y))])
    cmap2 = plt.cm.Set1

    # Plot regions
    if x1range is not None:
        x1_min, x1_max = x1range[0], x1range[1] + (x1range[1] - x1range[0]) / gridpoints
    else:
        x1_min, x1_max = X[:, x1].min(), X[:, x1].max() + (X[:, x1].max() - X[:, x1].min()) / gridpoints

    if x2range is not None:
        x2_min, x2_max = x2range[0], x2range[1] + (x2range[1] - x2range[0]) / gridpoints
    else:
        x2_min, x2_max = X[:, x2].min(), X[:, x2].max() + (X[:, x2].max() - X[:, x2].min()) / gridpoints

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, (x1_max - x1_min) / gridpoints),
                           np.arange(x2_min, x2_max, (x2_max - x2_min) / gridpoints))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape).round()

    plt.figure(1, figsize=figsize)

    # plt.contourf(xx1, xx2, Z, alpha=1, cmap=cmap1)
    plt.pcolormesh(xx1, xx2, Z, cmap=cmap1, shading='auto')
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot training points
    plt.scatter(X[:, x1], X[:, x2], c=y, edgecolors='k', cmap=cmap2)


# -----------------------------------------------------------

def plot_nv_1d(mu=0.0, var=1.0, xlim=(-5, 5), nx=100):
    x = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / nx)
    y = 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-1.0 / 2.0 * np.square(x - mu) / var)
    plt.plot(x, y)


# -----------------------------------------------------------

def plot_nv_3D(mu, Sigma, N=50, xlim=None, ylim=None, zlim=None, zticks=None, figsize=(5, 5)):
    if xlim is not None:
        X = np.linspace(xlim[0], xlim[1], N)
    else:
        X = np.linspace(-5, 5, N)

    if ylim is not None:
        Y = np.linspace(ylim[0], ylim[1], N)
    else:
        Y = np.linspace(-5, 5, N)

    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    F = multivariate_normal(mu, Sigma, True)
    Z = F.pdf(pos)

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True,
                    cmap=cm.inferno)

    cset = ax.contour(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.inferno)

    # Adjust the limits, ticks and view angle
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(xlim)

    ax.set_zlim(-0.15, Z.max())
    # ax.set_zticks(np.linspace(0, Z.max(), 5))
    ax.view_init(27, 300)


# -----------------------------------------------------------

def plot_nv_contour(mu, Sigma, N=50, xlim=None, ylim=None):
    if xlim is not None:
        X = np.linspace(xlim[0], xlim[1], N)
    else:
        X = np.linspace(-5, 5, N)

    if ylim is not None:
        Y = np.linspace(ylim[0], ylim[1], N)
    else:
        Y = np.linspace(-5, 5, N)

    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    F = multivariate_normal(mu, Sigma, True)
    Z = F.pdf(pos)

    cset = plt.contour(X, Y, Z, cmap=cm.inferno)

    # Adjust the limits, ticks and view angle
    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(xlim)


# -----------------------------------------------------------
def plot_history(hist,
                 figsize=(8, 5),
                 ylim=(0, None),
                 combined_plot=True,
                 loss_only=False,
                 acc_only=False,
                 title=None,
                 filename=None):
    """
    Plots the loss and/or accuracy values returned by a call to fit()

    Parameters
    ----------
    hist: the hist object returned by the fit() function

    figsize: list or tuple with 2 values
        size for the figure
    ylim: list or tuple with 2 values
        Min and max value for the y axis
    combined_plot: True: plot all curves in a combined plot
                   False: use separate plots
    loss_only: True: Only loss will be plotted
    acc_only: True: Only accuracy will be plotted
    title: title of the plot
    filename: filename for saving the plot

    Returns
    -------
    ---
    """

    legend = []

    # Make sure, that a y axis is labeled for plots with just one curve
    if len(hist.history.keys()) == 1: combined_plot = False

    if combined_plot: plt.figure(figsize=figsize)

    for m in hist.history.keys():  # hist.params['metrics']:

        if loss_only and 'loss' not in m: continue
        if acc_only and 'acc' not in m: continue

        if not combined_plot: plt.figure(figsize=figsize)

        y = hist.history[m]
        legend.append(m)

        # x_ticks = np.linspace(1, len(y), len(y))
        x_ticks = np.array(hist.epoch) + 1

        plt.plot(x_ticks, y)

        if not combined_plot:
            plt.xlabel('epoch')
            if len(y) <= 10:
                plt.xticks(x_ticks)

            plt.ylabel(m)
            plt.ylim(ylim)
            if title is not None:
                plt.title(title)
            if filename is not None:
                plt.savefig(filename)
            plt.show()

    if combined_plot:
        plt.xlabel('epoch')
        if len(y) <= 10:
            plt.xticks(x_ticks)
        plt.ylabel('')
        plt.ylim(ylim)
        plt.legend(legend)
        if title is not None:
            plt.title(title)
        if filename is not None:
            plt.savefig(filename)
        plt.show()


# -----------------------------------------------------------
def plot_accuracy(hist, figsize=(8, 5), ylim=(0, 1), title=None):
    plot_history(hist, figsize, ylim, title=title, acc_only=True)


# -----------------------------------------------------------
def plot_loss(hist, figsize=(8, 5), ylim=(0, 1), title=None):
    plot_history(hist, figsize, ylim, title=title, loss_only=True)
