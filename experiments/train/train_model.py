import argparse
import torch
from torch import distributions
from torch.utils.data import DataLoader

from data.datasets import ShapeNet
from utils.losses import loss_fun
from utils.MDS import multiDS
from models.models import model_init, model_load
from models.flows import F_flow, G_flow

import datetime
import yaml
import os


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_z = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
    prior_e = distributions.MultivariateNormal(torch.zeros(config['emb_dim']), torch.eye(config['emb_dim']))

    cloud = ShapeNet(config)
    dataloader = DataLoader(cloud, batch_size=config['batch_size'], shuffle=True)

    if not os.path.exists(config['save_models_dir']):
        os.makedirs(config['save_models_dir'])
    if not os.path.exists(config['save_models_dir_backup']):
        os.makedirs(config['save_models_dir_backup'])

    if config['load_models']:
        F_flows, G_flows, optimizer, scheduler, w = model_load(config, device)
    else:
        if config['init_w']:
            clouds = ShapeNet(config, mixed=False)
            w = multiDS(torch.from_numpy(clouds.cloud).float().to(device).contiguous(), config['emb_dim'], use_EMD=False).to(device).float()
            torch.save(w, config['save_models_dir'] + 'w.pth')
        else:
            w = torch.load(config['load_models_dir'] + 'w.pth')

        F_flows, G_flows, optimizer, scheduler = model_init(config, device)

    for key in F_flows:
        F_flows[key].train()
    for key in G_flows:
        G_flows[key].train()

    for i in range(config['n_epochs']):
        loss_acc = 0
        for j, (x, targets) in enumerate(dataloader):
            x = (x.float() + 1e-4 * torch.rand(x.shape)).to(device)
            w_iter = w[targets] + 1e-4 * torch.randn(w[targets].shape).to(device)

            e, e_ldetJ = G_flow(w_iter, G_flows, config['n_flows_G'], config['emb_dim'])
            z, z_ldetJ = F_flow(x, e, F_flows, config['n_flows_F'])

            loss = loss_fun(z, z_ldetJ, prior_z, e, e_ldetJ, prior_e)
            loss_acc += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (i + 1) % 2 == 0:
            path = config['save_models_dir_backup']
        else:
            path = config['save_models_dir']
            print('model_saved')

        for key in F_flows:
            torch.save(F_flows[key].state_dict(), path + 'F_' + key + '.pth')
        for key in G_flows:
            torch.save(G_flows[key].state_dict(), path + 'G_' + key + '.pth')
        torch.save(optimizer.state_dict(), path + 'optimizer.pth')
        torch.save(scheduler.state_dict(), path + 'scheduler.pth')

        with open(config['losses'], 'a') as file:
            file.write('Epoch: {}/{} Loss: {:.4f} Time: {}\n'.format(i + 1, config['n_epochs'], loss_acc / (j + 1),
                                                                     str(datetime.datetime.now().time()).split('.')[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
