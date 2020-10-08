import yaml
import argparse
import os
import torch
from torch import distributions
import tqdm
import numpy as np

from models.models import model_load
from models.flows import G_flow
from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds
from models.pointnet import Encoder


def nll(e, e_ldetJ, prior_e):
        ll_e = prior_e.log_prob(e.cpu()).to(e.device) + e_ldetJ
        return -ll_e


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_flows = model_load(config, device, train=False)[1]

    prior_e = distributions.MultivariateNormal(
            torch.zeros(config["emb_dim"]), torch.eye(config["emb_dim"])
        )

    pointnet = Encoder(
        load_pretrained=config["load_pretrained"],
        pretrained_path=config["pretrained_path"],
        zdim=config["emb_dim"],
    ).to(device)

    pointnet.load_state_dict(
            torch.load(os.path.join(config["load_models_dir"], "pointnet.pth"))
    )

    for key in G_flows:
        G_flows[key].eval()
    pointnet.eval()

    if config['use_random_dataloader']:
        tr_sample_size = 1
        te_sample_size = 1
    else:
        tr_sample_size = config['tr_sample_size']
        te_sample_size = config['te_sample_size']

    cloud_pointflow = ShapeNet15kPointClouds(
        tr_sample_size=tr_sample_size,
        te_sample_size=te_sample_size,
        root_dir=config["root_dir"],
        root_embs_dir=config["root_embs_dir"],
        normalize_per_shape=config["normalize_per_shape"],
        normalize_std_per_axis=config["normalize_std_per_axis"],
        split="train",
        scale=config["scale"],
        categories=config["categories"],
        random_subsample=True,
    )

    if config['use_random_dataloader']:
        cloud_pointflow = CIFDatasetDecorator(cloud_pointflow)

    if (
            config["resume_dataset_mean"] is not None
            and config["resume_dataset_std"] is not None
    ):
        mean = np.load(config["resume_dataset_mean"])
        std = np.load(config["resume_dataset_std"])
        cloud_pointflow.renormalize(mean, std)

        mean = torch.from_numpy(mean).to(device)
        std = torch.from_numpy(std).to(device)
    
    nlls = []
    for cloud in tqdm.tqdm(cloud_pointflow.all_points):
        cloud = torch.from_numpy(cloud[None, :]).to(device)
        with torch.no_grad():
            w = pointnet(cloud)
            e, ldetJ = G_flow(w, G_flows, config['n_flows_G'], config['emb_dim'])
            loss = nll(e, ldetJ, prior_e)
            nlls.append(loss.item())
    
    nlls = torch.tensor(nlls)
    rare_ids = torch.topk(nlls, 15, largest=True, sorted=True)[1]
    common_ids = torch.topk(nlls, 15, largest=False, sorted=True)[1]

    rare_clouds = cloud_pointflow.all_points[rare_ids, :2048, :]
    common_clouds = cloud_pointflow.all_points[common_ids, :2048, :]
    print(rare_clouds.shape, common_clouds.shape)
    torch.save(rare_clouds, config['load_models_dir'] + 'rare_clouds.pth')
    torch.save(common_clouds, config['load_models_dir'] + 'common_clouds.pth')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))