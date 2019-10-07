import argparse
import torch
import yaml
from utils.plotting_tools import plot_points
from models.flows import F_inv_flow
from models.models import model_load


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows, G_flows, _, _, _ = model_load(config, device, train=False)

    for key in F_flows:
        F_flows[key].eval()
    for key in G_flows:
        G_flows[key].eval()

    embs4g = torch.randn(config['n_samples'], config['emb_dim']).to(device)

    for l in range(config['n_samples']):
        z = torch.randn(config['n_points'], 3).to(device).float()

        with torch.no_grad():
            targets = torch.LongTensor(config['n_points'], 1).fill_(l)
            embeddings4g = embs4g[targets].view(-1, config['emb_dim'])

            z = F_inv_flow(z, embeddings4g, F_flows, config['n_flows_F'])

        plot_points(z.cpu().numpy(), config, save_name='gen_' + str(l), show=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
