import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_points(cloud, config, xlim=None, ylim=None, zlim=None, save_name=None, s=10, show=True):
    '''
    ploting point cloud
    '''
    if not os.path.exists(config['plots_dir']):
        os.makedirs(config['plots_dir'])

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
