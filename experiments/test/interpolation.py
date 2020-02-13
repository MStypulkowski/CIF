import os
import argparse
import torch
import yaml
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds
from models.models import model_load
from models.flows import G_flow_new, G_flow, F_inv_flow_new, F_inv_flow
from utils.plotting_tools import get_rotation_matrix


def update_cloud(idx, samples, plot, ax):
    xs = samples[idx, :, 0]
    ys = samples[idx, :, 1]
    zs = samples[idx, :, 2]
    plot[0].remove()
    plot[0] = ax.scatter(xs, ys, zs, c='darkblue', s=10)
    ax.set_xlim(np.quantile(xs, 0.05)-0.02, np.quantile(xs, 0.95)+0.02)
    ax.set_ylim(np.quantile(ys, 0.05)-0.02, np.quantile(ys, 0.95)+0.02)
    ax.set_zlim(np.quantile(zs, 0.05)-0.02, np.quantile(zs, 0.95)+0.02)


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows, G_flows, _, _, w = model_load(config, device, train=False)

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
        normalize_per_shape=config["normalize_per_shape"],
        normalize_std_per_axis=config["normalize_std_per_axis"],
        split="train",
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

    for key in F_flows:
        F_flows[key].eval()
    for key in G_flows:
        G_flows[key].eval()

    mean = (
        torch.from_numpy(test_cloud.all_points_mean)
            .float()
            .to(device)
            .squeeze(dim=0)
    )
    std = (
        torch.from_numpy(test_cloud.all_points_std)
            .float()
            .to(device)
            .squeeze(dim=0)
    )

    rotation_matrix = get_rotation_matrix([-np.pi / 2, 0., np.pi])

    samples = []
    for start, stop in zip(config['start_ids'], config['stop_ids']):

        w4inter = w[torch.tensor([start, stop])]

        with torch.no_grad():
            # map w into e
            if config['use_new_g']:
                embs4inter, _ = G_flow_new(w4inter, G_flows, config['n_flows_G'])
            else:
                embs4inter, _ = G_flow(w4inter, G_flows, config['n_flows_G'], config['emb_dim'])

            # create some embeddings between the two to interpolate
            n_midsamples = config['n_midsamples']
            step_vector = (embs4inter[1] - embs4inter[0]) / (n_midsamples + 1)

            embs_samples = [embs4inter[0]]
            for i in range(n_midsamples + 1):
                embs_samples.append(embs4inter[0] + i * step_vector)
            embs_samples = torch.stack(embs_samples)

            # generate samples
            for sample_index in tqdm.trange(n_midsamples + 2, desc="Sample"):
                z = config['prior_z_var'] * torch.randn(config['n_points'], 3).to(device).float()
                targets = torch.LongTensor(config['n_points'], 1).fill_(sample_index)
                embeddings4inter = embs_samples[targets].view(-1, config['emb_dim'])

                if config['use_new_f']:
                    z = F_inv_flow_new(z, embeddings4inter, F_flows, config['n_flows_F'])
                else:
                    z = F_inv_flow(z, embeddings4inter, F_flows, config['n_flows_F'])

                z = z * std + mean

                z_rotated = torch.from_numpy(np.dot(z.cpu().numpy(), rotation_matrix))
                samples.append(z_rotated)

    samples = torch.cat(samples, 0).view(-1, config['n_points'], 3)
    torch.save(samples, config['load_models_dir'] + 'interpolation_samples.pth')
    samples = np.array(samples)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if not os.path.exists(config["plots_dir"]):
        os.makedirs(config["plots_dir"])

    plot = [ax.scatter(samples[0, :, 0], samples[0, :, 1], samples[0, :, 2])]
    anim = animation.FuncAnimation(fig, update_cloud, fargs=(samples, plot, ax), frames=samples.shape[0], interval=50)
    anim.save(config['plots_dir'] + r'clouds' + str(config['n_midsamples']) + str(config['n_points']) + '.gif', writer='imagemagick')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
