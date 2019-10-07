import argparse
import torch
import yaml
import numpy as np
from models.models import model_load
from models.flows import G_flow, F_inv_flow
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


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

    for key in F_flows:
        F_flows[key].eval()
    for key in G_flows:
        G_flows[key].eval()

    samples = []
    for start, stop in zip(config['start_ids'], config['stop_ids']):

        w4inter = w[torch.tensor([start, stop])]

        with torch.no_grad():
            # map w into e
            embs4inter, _ = G_flow(w4inter, G_flows, config['n_flows_G'], config['emb_dim'])

            # create some embeddings between the two to interpolate
            n_midsamples = config['n_midsamples']
            step_vector = (embs4inter[1] - embs4inter[0]) / (n_midsamples + 1)

            embs_samples = [embs4inter[0]]
            for i in range(n_midsamples + 1):
                embs_samples.append(embs4inter[0] + i * step_vector)
            embs_samples = torch.stack(embs_samples)

            # generate samples
            for l in range(n_midsamples + 2):
                z = torch.randn(config['n_points'], 3).to(device).float()
                targets = torch.LongTensor(config['n_points'], 1).fill_(l)
                embeddings4inter = embs_samples[targets].view(-1, config['emb_dim'])

                z = F_inv_flow(z, embeddings4inter, F_flows, config['n_flows_F'])

                samples.append(z.cpu().numpy())
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

    plot = [ax.scatter(samples[0, :, 0], samples[0, :, 1], samples[0, :, 2])]
    anim = animation.FuncAnimation(fig, update_cloud, fargs=(samples, plot, ax), frames=samples.shape[0], interval=50)
    anim.save(config['plots_dir'] + r'clouds' + str(config['n_midsamples']) + str(config['n_points']) + '.mp4', writer='ffmpeg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
