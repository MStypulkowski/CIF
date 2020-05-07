import argparse
import torch
import tqdm
from collections import OrderedDict
from torch import distributions
from torch.utils.data import DataLoader
from models.architecture import W4Recon
from data.datasets_pointflow import ShapeNet15kPointClouds, CIFDatasetDecorator
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
        batch_size = config['batch_size_if_random_split']
    else:
        tr_sample_size = config['tr_sample_size']
        te_sample_size = config['te_sample_size']
        batch_size = config['batch_size']

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

    if config['use_random_dataloader']:
        test_cloud = CIFDatasetDecorator(test_cloud)

    dataloader_pointflow = DataLoader(
        test_cloud, batch_size=batch_size, shuffle=True
    )

    # n_test_clouds, cloud_size, _ = test_cloud.all_points.shape

    F_flows, G_flows, _, _, _ = model_load(config, device, train=False)

    w4recon = W4Recon(config).to(device)
    optimizer4recon = torch.optim.Adam(w4recon.parameters(), lr=config['l_rate4recon'])
    scheduler4recon = torch.optim.lr_scheduler.StepLR(optimizer4recon, step_size=10, gamma=0.8)

    if not os.path.exists(config['save_models_dir']):
        os.makedirs(config['save_models_dir'])

    if config['load_embs']:
        w4recon.load_state_dict(torch.load(config['save_models_dir'] + r'embs.pth'))
        optimizer4recon.load_state_dict(torch.load(config['save_models_dir'] + r'embs_optimizer.pth'))
        scheduler4recon.load_state_dict(torch.load(config['save_models_dir'] + r'embs_scheduler.pth'))

    # data = (torch.tensor(test_cloud.cloud[config['id4recon']]).float()).to(device)
    # data = torch.tensor(test_cloud.all_points).view(-1, 3).to(device)
    # targets = torch.LongTensor(cloud_size, 1).fill_(0)
    # idx_w = [i for i in range(n_test_clouds) for j in range(cloud_size)]
    # data_batches = [data[:len(data)//2], data[len(data)//2:]]
    # idx_batches = [idx_w[:len(idx_w)//2], idx_w[len(idx_w)//2:]]

    # freeze flows
    for key in G_flows:
        G_flows[key].eval()
    for key in F_flows:
        F_flows[key].eval()
    w4recon.train()

    # loss_acc_z = 0.
    # loss_acc_e = 0.
    # pbar = tqdm.tqdm(dataloader_pointflow, desc="Batch")
    for i in range(config['n_epochs']):
        print("Epoch: {} / {}".format(i + 1, config["n_epochs"]))

        loss_acc_z = 0.
        loss_acc_e = 0.
        pbar = tqdm.tqdm(dataloader_pointflow, desc="Batch")
        for j, datum in enumerate(pbar):
            idx_batch, tr_batch, te_batch = (
                datum["idx"],
                datum["train_points"],
                datum["test_points"],
            )

            if not config['use_random_dataloader']:
                b, n_points, coords_num = tr_batch.shape
                idx_batch = (
                    idx_batch.unsqueeze(dim=-1)
                        .repeat(repeats=(1, n_points))
                        .reshape((-1,))
                )

            tr_batch = (
                (tr_batch.float() + config['x_noise'] * torch.rand(tr_batch.shape))
                    .to(device)
                    .reshape((-1, 3))
            )

            w = w4recon()
            # print(w[0])
            w_iter = w[idx_batch] + config['w_noise'] * torch.randn(w[idx_batch].shape).to(
                device
            )
            # print(w_iter.shape)
            # w_iter = w[idx_batch]
            e, e_ldetJ = G_flow(w_iter, G_flows, config['n_flows_G'], config['emb_dim'])
            # print(e.shape)
            z, z_ldetJ = F_flow(tr_batch, e, F_flows, config['n_flows_F'])
            # print(z.shape)

            loss_z, loss_e = loss_fun(z, z_ldetJ, prior_z, e, e_ldetJ, prior_e, _lambda=config['e_loss_scale'])
            loss = loss_e + loss_z
            loss_acc_e += loss_e.item()
            loss_acc_z += loss_z.item()

            optimizer4recon.zero_grad()
            loss.backward()
            optimizer4recon.step()

            pbar.set_postfix(
                OrderedDict(
                    {
                        "loss_e": "%.4f" % (loss_e.item()),
                        "loss_e_avg": "%.4f" % (loss_acc_e / (j + 1)),
                        "loss_z": "%.4f" % (loss_z.item()),
                        "loss_z_avg": "%.4f" % (loss_acc_z / (j + 1)),
                        "loss": "%.4f" % (loss_e.item() + loss_z.item()),
                        "loss_avg": "%.4f" % ((loss_acc_z + loss_acc_e) / (j + 1))
                    }
                )
            )
        scheduler4recon.step()

        # save embs and gradients
        torch.save(w4recon.state_dict(), config['save_models_dir'] + r'embs.pth')
        torch.save(optimizer4recon.state_dict(), config['save_models_dir'] + r'embs_optimizer.pth')
        torch.save(scheduler4recon.state_dict(), config['save_models_dir'] + r'embs_scheduler.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
