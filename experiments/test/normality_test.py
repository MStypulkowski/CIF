import argparse
import torch
import yaml
from models.flows import G_flow_new, G_flow
from models.models import model_load
from scipy.stats import normaltest
from data.datasets_pointflow import ShapeNet15kPointClouds
import matplotlib.pyplot as plt
import os
import seaborn as sns


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, G_flows, _, _ = model_load(config, device, train=False)

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

    for key in G_flows:
        G_flows[key].eval()

    w = torch.from_numpy(test_cloud.all_ws).to(device)

    with torch.no_grad():
        if config['use_new_g']:
            e, _ = G_flow_new(w, G_flows, config['n_flows_G'])
        else:
            e, _ = G_flow(w, G_flows, config['n_flows_G'], config['emb_dim'])
    print(e[:, 0].shape)
    print(torch.mean(e, dim=0).shape)
    means, stds = torch.mean(e, dim=0), torch.std(e, dim=0)

    for i, (mean, std) in enumerate(zip(means, stds)):
        _, p_val = normaltest(e[:, i].cpu())
        plt.figure()
        plt.hist(e[:, i].cpu(), bins=50)

        if not os.path.exists(config['load_models_dir'] + 'histograms/'):
            os.makedirs(config['load_models_dir'] + 'histograms/')

        plt.savefig(config['load_models_dir'] + 'histograms/hist' + str(i) + '.png')
        plt.close()

        plt.figure()
        sns.boxplot(x="x", y="y", data={
            "x": ["Dim"] * e.shape[0],
            "y": e[:, i].cpu()
        })
        plt.savefig(config['load_models_dir'] + 'histograms/box' + str(i) + '.png')
        plt.close()

        print(f'Dim {i}: mean: {mean:.2f} std: {std:.2f} p_val: {p_val:.4f}')

        if p_val >= 0.05:
            print('True')
    print('\n')
    print('Mean of means: {:.4f} std of means: {:.4f}'.format(torch.mean(means).item(), torch.std(means).item()))
    print('Mean of stds: {:.4f} std of stds: {:.4f}'.format(torch.mean(stds).item(), torch.std(stds).item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
