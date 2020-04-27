import argparse
import torch
import yaml
import numpy as np
from utils.metrics import MMD, pairwise_MMD
from models.models import model_load
from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows, _, _, _ = model_load(config, device, train=False)

    if config['use_random_dataloader']:
        tr_sample_size = 1
        te_sample_size = 1
    else:
        tr_sample_size = config['tr_sample_size']
        te_sample_size = config['te_sample_size']

    test_cloud = ShapeNet15kPointClouds(
        tr_sample_size=tr_sample_size,
        te_sample_size=te_sample_size,
        root_dir=config["root_dir"],
        root_embs_dir=config["root_embs_dir"],
        normalize_per_shape=config["normalize_per_shape"],
        normalize_std_per_axis=config["normalize_std_per_axis"],
        split="val",
        scale=config["scale"],
        categories=config["categories"],
        random_subsample=True,
    )

    if config['use_random_dataloader']:
        test_cloud = CIFDatasetDecorator(test_cloud)

    if (
            config["resume_dataset_mean"] is not None
            and config["resume_dataset_std"] is not None
    ):
        mean = np.load(config["resume_dataset_mean"])
        std = np.load(config["resume_dataset_std"])
        test_cloud.renormalize(mean, std)

        mean = torch.from_numpy(mean).to(device)
        std = torch.from_numpy(std).to(device)

    for key in F_flows:
        F_flows[key].eval()

    n_samples = test_cloud.all_points.shape[0]
    cloud_size = 2048

    print(n_samples, cloud_size)

    samples = torch.load(config['load_models_dir'] + 'test_recon_samples.pth').to(device)
    ref_samples = torch.from_numpy(test_cloud.all_points[:, :2048, :]).float().to(device)
    ref_samples = ref_samples * std + mean
    print(ref_samples.shape)

    if config["use_EMD"]:
        print("MMD (EMD): {:.8f}".format(MMD(samples, ref_samples).item()))
        print("pairwise MMD (EMD): {:.8f}".format(pairwise_MMD(samples, ref_samples).item()))

    else:
        print("MMD (CD): {:.8f}".format(MMD(samples, ref_samples, use_EMD=False).item()))
        print("pairwise MMD (CD): {:.8f}".format(pairwise_MMD(samples, ref_samples, use_EMD=False).item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
