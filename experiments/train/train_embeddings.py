import argparse
import torch
import tqdm
from collections import OrderedDict
from torch import distributions
from models.architecture import W4Recon
from data.datasets_pointflow import ShapeNet15kPointClouds
from models.flows import F_flow, G_flow
import yaml
from utils.losses import loss_fun
from models.models import model_load
import os


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_z = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
    prior_e = distributions.MultivariateNormal(torch.zeros(config['emb_dim']),
                                               torch.eye(config['emb_dim']))

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
        split="val",
        scale=config["scale"],
        categories=config["categories"],
        random_subsample=True,
    )

    n_test_clouds, cloud_size, _ = test_cloud.cloud.shape

    F_flows, G_flows, _, _, _ = model_load(config, device, train=False)

    w4recon = W4Recon(config).to(device)
    optimizer4recon = torch.optim.Adam(w4recon.parameters(), lr=config['l_rate4recon'])
    scheduler4recon = torch.optim.lr_scheduler.StepLR(optimizer4recon, step_size=400, gamma=0.8)

    if not os.path.exists(config['embs_dir']):
        os.makedirs(config['embs_dir'])

    if config['load_embs']:
        w4recon.load_state_dict(torch.load(config['embs_dir'] + r'embs.pth'))
        optimizer4recon.load_state_dict(torch.load(config['embs_dir'] + r'optimizer.pth'))
        scheduler4recon.load_state_dict(torch.load(config['embs_dir'] + r'scheduler.pth'))

    # data = (torch.tensor(test_cloud.cloud[config['id4recon']]).float()).to(device)
    data = test_cloud.all_points.to(device)
    # targets = torch.LongTensor(cloud_size, 1).fill_(0)

    # freeze flows
    for key in G_flows:
        G_flows[key].eval()
    for key in F_flows:
        F_flows[key].eval()
    w4recon.train()

    loss_acc_z = 0.
    loss_acc_e = 0.
    pbar = tqdm.trange(config['n_epochs'])
    for i in pbar:
        # noise = torch.randn(test_cloud.all_points.shape).to(device)
        # x = data + 1e-4 * noise
        # embeddings4recon = embs4recon(targets).view(-1, config['emb_dim'])

        # x, z_ldetJ = F_flow(x, embeddings4recon, F_flows, config['n_flows_F'])

        e, e_ldetJ = G_flow(w4recon(), G_flows, config['n_flows_G'], config['emb_dim'])
        z, z_ldetJ = F_flow(data, e, F_flows, config['n_flows_F'])

        loss_z, loss_e = loss_fun(z, z_ldetJ, prior_z, e, e_ldetJ, prior_e, _lambda=config['e_loss_scale'])
        loss = loss_e + loss_z
        loss_acc_e += loss_e.item()
        loss_acc_z += loss_z.item()

        optimizer4recon.zero_grad()
        loss.backward()
        optimizer4recon.step()
        scheduler4recon.step()

        pbar.set_postfix(
            OrderedDict(
                {
                    "loss_e": "%.4f" % (loss_e.item()),
                    "loss_e_avg": "%.4f" % (loss_acc_e / (i + 1)),
                    "loss_z": "%.4f" % (loss_z.item()),
                    "loss_z_avg": "%.4f" % (loss_acc_z / (i + 1)),
                    "loss": "%.4f" % (loss_e.item() + loss_z.item()),
                    "loss_avg": "%.4f" % ((loss_acc_z + loss_acc_e) / (i + 1))
                }
            )
        )

        if (i + 1) % 50 == 0:
            # print('Epoch: {}/{} Loss: {:.4f} l_rate: {:.4f}'.format(i + 1, config['n_epochs'], loss.item(), scheduler4recon.get_lr()[0]))

            # save embs and gradients
            torch.save(w4recon.state_dict(), config['embs_dir'] + r'embs.pth')
            torch.save(optimizer4recon.state_dict(), config['embs_dir'] + r'optimizer.pth')
            torch.save(scheduler4recon.state_dict(), config['embs_dir'] + r'scheduler.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
