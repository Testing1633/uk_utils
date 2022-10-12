import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame
#from numbers import Integral
from sklearn.preprocessing import LabelEncoder


from scipy.stats import multivariate_normal

def get_colormap():
    return plt.cm.tab10


# -----------------------------------------------------------


def plot_decision_regions(X=None, y=None,
                          classifier=None,
                          data=None, features=None, classes=None,  # for DataFrames only
                          x1range=None, x2range=None,
                          x1=0, x2=1,
                          gridpoints=100,
                          # figsize=(6, 6),
                          plot_legend=True,
                          ax=None
                          ):
    '''
    '''
    # Select colormaps
    # markers = ('s', 'x', 'o', '^', 'v')
    # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    if X is None and data is None and (x1range is None or x2range is None):
        print('ERROR: Cannot plot decision regions without specifying data or a range')
        return

    cmap1 = get_colormap()  # for the mesh
    cmap2 = get_colormap()  # for scatter plots

    # Collect input features from a pandas DataFrame
    if data is not None:
        if type(data) is not DataFrame:
            print('ERROR: Parameter data must be of type DataFrame')
            return

        # Features
        X = data[features].values
        feat_names = features

        # Class / target output
        y = data[classes].values
    else:
        feat_names = (f'$x_{x1+1}$', f'$x_{x2+1}$')

    # Use a label encoder. Does not change the encoding for int's.
    if y is None:
        y = np.zeros(X.shape[0], dtype=np.int8)

    class_encoder = LabelEncoder()
    y = class_encoder.fit_transform(y)
    class_names = class_encoder.classes_

    # Compute bounds/intervals for the color normalizer used by pcolormesh and scatter
    bounds = np.linspace(-0.5, len(class_names) - 0.5, len(class_names) + 1)

    # Set plot range for x- and y-direction
    if x1range is not None:
        x1_min, x1_max = x1range[0], x1range[1] + \
                         (x1range[1] - x1range[0]) / gridpoints
    else:
        dx = (X[:, x1].max() - X[:, x1].min()) / 20
        x1_min, x1_max = X[:, x1].min() - dx, X[:, x1].max() + dx

    if x2range is not None:
        x2_min, x2_max = x2range[0], x2range[1] + \
                         (x2range[1] - x2range[0]) / gridpoints
    else:
        dy = (X[:, x2].max() - X[:, x2].min()) / 20
        x2_min, x2_max = X[:, x2].min() - dy, X[:, x2].max() + dy

    # Plot the class
    if classifier is not None:
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, (x1_max - x1_min) / gridpoints),
                               np.arange(x2_min, x2_max, (x2_max - x2_min) / gridpoints))

        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        if Z.ndim > 1:
            Z = np.argmax(Z, axis=1)
        Z = class_encoder.transform(Z)
        Z = Z.reshape(xx1.shape).astype(np.int32)  # .round()

        plt.grid(False)
        plt.pcolormesh(xx1, xx2, Z,
                       norm=BoundaryNorm(boundaries=bounds, ncolors=len(class_names)),
                       alpha=0.3,
                       cmap=cmap1)

        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
    else:
        plt.xlim(x1_min, x1_max)
        plt.xlim(x2_min, x2_max)

    # Plot training points
    ax = plt.gca()
    if X is not None:
        if y is not None:
            scatter = ax.scatter(X[:, x1], X[:, x2], c=y, edgecolors='k', linewidth=0.5,
                                 norm=BoundaryNorm(boundaries=bounds, ncolors=len(class_names)),
                                 cmap=cmap2)
        else:
            scatter = ax.scatter(X[:, x1], X[:, x2], edgecolors='k', linewidth=0.5,
                                 norm=BoundaryNorm(boundaries=bounds, ncolors=len(class_names)),
                                 cmap=cmap2)

    legend1 = ax.legend(*(scatter.legend_elements()[0], class_names),
                        loc='best', title='Classes')
    if len(class_names) > 1:
        ax.add_artist(legend1)

    plt.xlabel(feat_names[0])
    plt.ylabel(feat_names[1])


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
def plot_loss(hist, figsize=(8, 5), ylim=(0, None), title=None):
    plot_history(hist, figsize, ylim, title=title, loss_only=True)
