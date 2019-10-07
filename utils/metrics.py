import torch
from utils.pytorch_structural_losses.metrics import chamfer_distance, earth_mover_distance


def MMD(samples, ref_clouds, use_EMD=True):
    '''
    for each cloud from ref_clouds find distance from nearest sample cloud
    returns mean of these distances
    '''
    with torch.no_grad():
        n_samples, n_points, _ = samples.shape

        if use_EMD:
            distance = earth_mover_distance
        else:
            distance = chamfer_distance

        min_dists = []

        for ref_cloud in ref_clouds:
            multi_ref = ref_cloud.expand(n_samples, n_points, 3).contiguous()
            dists = distance(samples, multi_ref)
            min_dist = torch.min(dists)
            min_dists.append(min_dist)

    return sum(min_dists) / len(min_dists)


def coverage(samples, ref_clouds, use_EMD=True):
    '''
    for each sample cloud find nearest ref_cloud
    returns |unique(nearest_ref_cloud)| / |ref_cloud|
    '''
    with torch.no_grad():
        n_refs, n_points, _ = ref_clouds.shape

        if use_EMD:
            distance = earth_mover_distance
        else:
            distance = chamfer_distance

        nearest_clouds = []

        for sample in samples:
            multi_sample = sample.expand(n_refs, n_points, 3).contiguous()
            dists = distance(multi_sample, ref_clouds)
            nearest_cloud = torch.argmin(dists)
            nearest_clouds.append(nearest_cloud)

    return len(torch.unique(torch.tensor(nearest_clouds))) / n_refs
