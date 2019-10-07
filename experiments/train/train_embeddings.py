import argparse
import torch
from torch import distributions
from models.architecture import Embeddings4Recon
from data.datasets import ShapeNet
from models.flows import F_flow
import yaml
from utils.losses import loss_fun_ret
from models.models import model_load
import os


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_z = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))

    test_cloud = ShapeNet(config, split='test', mixed=False)
    n_test_clouds, cloud_size, _ = test_cloud.cloud.shape

    F_flows, _, _, _, w = model_load(config, device, train=False)

    embs4recon = Embeddings4Recon(1, config['emb_dim']).to(device)
    optimizer4recon = torch.optim.Adam(embs4recon.parameters(), lr=config['l_rate4recon'])
    scheduler4recon = torch.optim.lr_scheduler.StepLR(optimizer4recon, step_size=400, gamma=0.8)

    if not os.path.exists(config['embs_dir']):
        os.makedirs(config['embs_dir'])

    path = config['embs_dir']

    if config['load_embs']:
        embs4recon.load_state_dict(torch.load(path + r'embs.pth'))
        optimizer4recon.load_state_dict(torch.load(path + r'optimizer.pth'))
        scheduler4recon.load_state_dict(torch.load(path + r'scheduler.pth'))

    data = (torch.tensor(test_cloud.cloud[config['id4recon']]).float()).to(device)
    targets = torch.LongTensor(cloud_size, 1).fill_(0)

    # freeze flows
    for key in F_flows:
        F_flows[key].eval()
    embs4recon.train()

    for i in range(config['n_epochs']):
        noise = torch.rand(test_cloud.cloud[config['id4recon']].shape).to(device)
        x = data + 1e-4 * noise
        embeddings4recon = embs4recon(targets).view(-1, config['emb_dim'])

        x, z_ldetJ = F_flow(x, embeddings4recon, F_flows, config['n_flows_F'])

        loss = loss_fun_ret(x, z_ldetJ, prior_z)

        optimizer4recon.zero_grad()
        loss.backward()
        optimizer4recon.step()
        scheduler4recon.step()
        if (i + 1) % 50 == 0:
            print('Epoch: {}/{} Loss: {:.4f} l_rate: {:.4f}'.format(i + 1, config['n_epochs'], loss.item(), scheduler4recon.get_lr()[0]))

            # save embs and gradients
            torch.save(embs4recon.state_dict(), path + r'embs.pth')
            torch.save(optimizer4recon.state_dict(), path + r'optimizer.pth')
            torch.save(scheduler4recon.state_dict(), path + r'scheduler.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
