import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def get_rotation_matrix(angles):
    x, y, z = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])
    return np.dot(Rx, np.dot(Ry, Rz))


def plot_points(cloud, config, xlim=None, ylim=None, zlim=None, save_name=None, s=10, show=True):
    '''
    ploting point cloud
    '''
    if not os.path.exists(config['plots_dir']):
        os.makedirs(config['plots_dir'])

    rotation_matrix = get_rotation_matrix([-np.pi/2, 0., np.pi])
    cloud = np.dot(cloud, rotation_matrix)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=s)
    if xlim:
        ax.set_xlim([-xlim, xlim])
    if ylim:
        ax.set_ylim([-ylim, ylim])
    if zlim:
        ax.set_zlim([-zlim, zlim])
    if save_name:
        plt.savefig(config['plots_dir'] + save_name + '.png', bbox_inches='tight')
    if show:
        plt.show()
