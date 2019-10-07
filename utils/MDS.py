import numpy as np
import torch
from sklearn.manifold import MDS
from utils.pytorch_structural_losses.metrics import chamfer_distance, earth_mover_distance


def multiDS(clouds, emb_dim, use_EMD=True):
    n_clouds, n_points, _ = clouds.shape
    if use_EMD:
        distance = earth_mover_distance
    else:
        distance = chamfer_distance

    dists = []
    for i in range(n_clouds):
        if i % 100 == 0:
            print('Calculating {}/{} distance'.format(i, n_clouds))
        cloud_dist = []
        for j in range(n_clouds):
            if i == j:
                continue
            dist = distance(clouds[i], clouds[j]).item()
            cloud_dist.append(dist)
        dists.append(cloud_dist)
    dists = np.array(dists)
    print('Distance matrix shape: ' + str(dists.shape))
    mds = MDS(n_components=emb_dim)
    w = mds.fit_transform(dists)

    return torch.from_numpy(w)
