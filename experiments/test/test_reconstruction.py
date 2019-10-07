import argparse
import torch
from models.architecture import Embeddings4Recon
from data.datasets import ShapeNet
import yaml
from models.models import model_load
from utils.plotting_tools import plot_points
from models.flows import F_inv_flow


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_cloud = ShapeNet(config, split='test', mixed=False)
    n_test_clouds, cloud_size, _ = test_cloud.cloud.shape

    F_flows, _, _, _, w = model_load(config, device, train=False)
    embs4recon = Embeddings4Recon(1, config['emb_dim']).to(device)

    embs4recon.load_state_dict(torch.load(config['embs_dir'] + r'embs.pth'))

    data = (torch.tensor(test_cloud.cloud[config['id4recon']]).float()).to(device)

    for key in F_flows:
        F_flows[key].eval()
    embs4recon.eval()

    z = torch.randn(config['n_points'], 3).to(device).float()
    with torch.no_grad():
        targets = torch.LongTensor(config['n_points'], 1).fill_(0)
        embeddings = embs4recon(targets).view(-1, config['emb_dim'])

        z = F_inv_flow(z, embeddings, F_flows, config['n_flows_F'])

    plot_points(z.cpu().numpy(), config, save_name='test_recon_' + str(config['id4recon']), show=False)
    plot_points(data.cpu().numpy(), config, save_name='test_ref_' + str(config['id4recon']), show=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
