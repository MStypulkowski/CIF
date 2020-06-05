import numpy as np
import torch
from sklearn.manifold import MDS
try:
    from utils.pytorch_structural_losses.metrics import chamfer_distance, earth_mover_distance
except:
    from utils.pytorch_structural_losses.metrics_cd import chamfer_distance


def multiDS(clouds, emb_dim, dists=None, use_EMD=True, load_dists=False):
    n_clouds, n_points, _ = clouds.shape
    print(n_clouds, n_points)
    if n_points > 5000:
        perm = torch.randperm(n_points)
        clouds = clouds[:, perm, :][:, :5000, :]

    print(f'Cloud shape: {clouds.shape}')

    if not load_dists:
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
                dist = distance(clouds[i], clouds[j]).item()
                cloud_dist.append(dist)
            dists.append(cloud_dist)
        dists = np.array(dists)
        print('Distance matrix shape: ' + str(dists.shape))

    mds = MDS(n_components=emb_dim, dissimilarity="precomputed", random_state=42)
    w = mds.fit_transform(dists)

    if not load_dists:
        return torch.from_numpy(dists), torch.from_numpy(w)
    return torch.from_numpy(w)
