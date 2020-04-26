import argparse
import yaml
import numpy as np
import torch
import tqdm
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.manifold import MDS

from utils.pytorch_structural_losses.metrics import chamfer_distance
from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds
# from sklearn import manifold
# from sklearn.metrics import pairwise

import scipy.optimize as sopt


# def get_rotation_matrix(angles):
#     x, y, z = angles
#     Rx = np.array([[1, 0, 0],
#                    [0, np.cos(x), -np.sin(x)],
#                    [0, np.sin(x), np.cos(x)]])
#     Ry = np.array([[np.cos(y), 0, np.sin(y)],
#                    [0, 1, 0],
#                    [-np.sin(y), 0, np.cos(y)]])
#     Rz = np.array([[np.cos(z), -np.sin(z), 0],
#                    [np.sin(z), np.cos(z), 0],
#                    [0, 0, 1]])
#     return np.dot(Rx, np.dot(Ry, Rz))


# def get_dists(clouds):
#     n_clouds, n_points, _ = clouds.shape
#     print(n_clouds, n_points)
#     perm = torch.randperm(n_points)
#     clouds = clouds[:, perm, :][:, :5000, :]
#
#     print(f'Cloud shape: {clouds.shape}')
#
#     distance = chamfer_distance
#
#     dists = []
#     for i in tqdm.trange(n_clouds, desc='Distance'):
#         # if i % 100 == 0:
#         #     print('Calculating {}/{} distance'.format(i, n_clouds))
#         cloud_dist = []
#         for j in range(n_clouds):
#             if i == j:
#                 continue
#             dist = distance(clouds[i], clouds[j]).item()
#             cloud_dist.append(dist)
#         dists.append(cloud_dist)
#     dists = np.array(dists)
#     return torch.from_numpy(dists)


def get_test_dists(test_clouds, clouds):
    n_clouds, n_points, _ = clouds.shape
    n_test, _, _ = test_clouds.shape
    # print(n_clouds, n_points)
    perm = torch.randperm(n_points)
    clouds = clouds[:, perm, :][:, :5000, :]
    test_clouds = test_clouds[:, perm, :][:, :5000, :]

    print(f'Cloud shape: {clouds.shape}, test cloud shape: {test_clouds.shape}')

    distance = chamfer_distance

    dists = []
    for i in tqdm.trange(n_test, desc='Distances'):
        # if i % 100 == 0:
        #     print('Calculating {}/{} distance'.format(i, n_test))
        cloud_dist = []
        for j in range(n_clouds):
            dist = distance(test_clouds[i], clouds[j]).item()
            cloud_dist.append(dist)
        dists.append(cloud_dist)
    dists = np.array(dists)
    return torch.from_numpy(dists)


def get_data(dir, category=['chair']):
    tr_sample_size = 1
    te_sample_size = 1

    cloud = CIFDatasetDecorator(
        ShapeNet15kPointClouds(
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            root_dir=dir,
            normalize_per_shape=False,
            normalize_std_per_axis=False,
            split="train",
            scale=1.0,
            categories=category,
            random_subsample=True,
        )
    )

    val_cloud = CIFDatasetDecorator(
        ShapeNet15kPointClouds(
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            root_dir=dir,
            normalize_per_shape=False,
            normalize_std_per_axis=False,
            split="val",
            scale=1.0,
            categories=category,
            random_subsample=True,
        )
    )
    return cloud, val_cloud


def embed_point(TD, W):
    """Embed a single point into the MDS space.

    Args:
        - TD: vector of length N of distances of the point to the N points in the MDS space
        - W: matrix N x D of locations of the N points in the D-dimensional MDS space
    Returns:
     vector of length D: the location of the point in the MDS space
    """
    assert TD.shape[0] == W.shape[0]
    assert len(TD.shape) == 1
    assert len(W.shape) == 2
    TD = TD.detach().double()
    W = W.detach().double()
    init_loc = W.mean(0, keepdim=True).numpy()

    def stress_fun(loc):
        loc = torch.from_numpy(loc).double()
        loc.requires_grad_(True)
        emb_TD = torch.sqrt(((W - loc) ** 2).sum(1)).double()
        stress = ((emb_TD - TD) ** 2).sum().double()
        stress.backward()
        return stress.item(), loc.grad.numpy()

    loc = sopt.fmin_l_bfgs_b(stress_fun, init_loc)[0]
    return torch.from_numpy(loc)


def embed_set(TD, W):
    ret = []
    for i in tqdm.trange(TD.shape[0], desc='Embed'):
        ret.append(embed_point(TD[i], W))
    return torch.stack(ret)


def get_w_test(config, device, return_matrix=True):
    cloud, val_cloud = get_data(config['root_dir'], config['categories'])
    test_dists = get_test_dists(torch.from_numpy(val_cloud.all_points).float().to(device),
                                    torch.from_numpy(cloud.all_points).float().to(device))
    torch.save(test_dists, config['load_models_dir'] + 'test_dists.pth')
    print('test_dists saved')

    w = torch.load(config['load_models_dir'] + 'w.pth')
    print('w loaded')

    print(w.shape, test_dists.shape)

    test_dists = test_dists.cpu()
    w = w.cpu()
    w_test = embed_set(test_dists, w)
    torch.save(w_test.float(), config['load_models_dir'] + 'w_test.pth')
    print('w_test saved')
    if return_matrix:
        return w_test


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_w_test(config, device, return_matrix=False)


if __name__ == '__main__':
    # print(f'is CUDA AVAILABLE {torch.cuda.is_available()}')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))