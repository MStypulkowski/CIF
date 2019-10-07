import argparse
import torch
import yaml
import numpy as np
from utils.metrics import MMD, coverage
from data.datasets import ShapeNet
from models.models import model_load
from models.flows import F_inv_flow


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows, _, _, _, w = model_load(config, device, train=False)

    test_cloud = ShapeNet(config, split='test', mixed=False)

    for key in F_flows:
        F_flows[key].eval()

    n_test_clouds, cloud_size, _ = test_cloud.cloud.shape
    n_samples = 3 * n_test_clouds

    samples = []
    embs4g = torch.randn(n_samples, config['emb_dim']).to(device)

    for l in range(n_samples):
        if l % 100 == 0:
            print('Generating {}/{} sample'.format(l, n_samples))
        z = torch.randn(cloud_size, 3).to(device).float()
        with torch.no_grad():
            targets = torch.LongTensor(cloud_size, 1).fill_(l)
            embeddings4g = embs4g[targets].view(-1, config['emb_dim'])

            z = F_inv_flow(z, embeddings4g, F_flows, config['n_flows_F'])

            samples.append(z.cpu().numpy())
    samples = np.array(samples).reshape(n_samples, cloud_size, 3)

    if config['use_EMD']:
        print('Coverage (EMD): {:.4f}%'.format(coverage(torch.from_numpy(samples).to(device), torch.from_numpy(test_cloud.cloud).float().to(device))*100))
        print('MMD (EMD): {:.4f}'.format(MMD(torch.from_numpy(samples).to(device), torch.from_numpy(test_cloud.cloud).float().to(device)).item()))

    else:
        print('Coverage (CD): {:.4f}%'.format(coverage(torch.from_numpy(samples).to(device), torch.from_numpy(test_cloud.cloud).float().to(device), use_EMD=False) * 100))
        print('MMD (CD): {:.4f}'.format(MMD(torch.from_numpy(samples).to(device), torch.from_numpy(test_cloud.cloud).float().to(device), use_EMD=False).item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
