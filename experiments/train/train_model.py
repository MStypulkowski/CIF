import argparse
import datetime
import os
import typing as t
import copy
from collections import OrderedDict

import torch
import tqdm
import yaml
from torch import distributions
from torch.utils.data import DataLoader


from data.datasets_pointflow import ShapeNet15kPointClouds, CIFDatasetDecorator
from models.flows import F_flow, G_flow
from models.models import model_init, model_load
from utils.MDS import multiDS
from utils.losses import loss_fun


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_z = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
    prior_e = distributions.MultivariateNormal(
        torch.zeros(config["emb_dim"]), torch.eye(config["emb_dim"])
    )

    # each dataset needs to be decorated
    cloud_pointflow = CIFDatasetDecorator(
        ShapeNet15kPointClouds(
            tr_sample_size=config["tr_sample_size"],
            te_sample_size=config["te_sample_size"],
            root_dir=config["root_dir"],
            normalize_per_shape=config["normalize_per_shape"],
            normalize_std_per_axis=config["normalize_std_per_axis"],
            split="train",
            scale=config["scale"],
            categories=config["categories"],

            # needed for compatibility, does not change behaviour of the class
            random_subsample=True,
        )
    )

    dataloader_pointflow = DataLoader(
        cloud_pointflow, batch_size=config["batch_size"], shuffle=True
    )

    if not os.path.exists(config["save_models_dir"]):
        os.makedirs(config["save_models_dir"])
    if not os.path.exists(config["save_models_dir_backup"]):
        os.makedirs(config["save_models_dir_backup"])

    if config["load_models"]:
        F_flows, G_flows, optimizer, scheduler, w = model_load(config, device)
    else:
        if config["init_w"]:
            # "all_points" -> [num_instances, num_points, coordinates]
            w = (
                multiDS(
                    torch.from_numpy(cloud_pointflow.all_points)
                    .float()
                    .to(device)
                    .contiguous(),
                    config["emb_dim"],
                    use_EMD=False,
                )
                .to(device)
                .float()
            )
            torch.save(w, config["save_models_dir"] + "w.pth")
        else:
            w = torch.load(config["load_models_dir"] + "w.pth")

        F_flows, G_flows, optimizer, scheduler = model_init(config, device)

    for key in F_flows:
        F_flows[key].train()
    for key in G_flows:
        G_flows[key].train()

    for i in range(config["n_epochs"]):
        print("Epoch: {} / {}".format(i + 1, config["n_epochs"]))

        loss_acc = 0
        pbar = tqdm.tqdm(dataloader_pointflow, desc="Batch")
        for j, datum in enumerate(pbar):
            # "idx_batch" -> indices of instances of a class
            idx_batch, tr_batch, te_batch = (
                datum["idx"],
                datum["train_points"],
                datum["test_points"],
            )

            tr_batch = (
                (tr_batch.float() + 1e-4 * torch.rand(tr_batch.shape))
                .to(device)
                .reshape((-1, 3))
            )
            w_iter = w[idx_batch] + 1e-4 * torch.randn(w[idx_batch].shape).to(
                device
            )

            e, e_ldetJ = G_flow(
                w_iter, G_flows, config["n_flows_G"], config["emb_dim"]
            )
            z, z_ldetJ = F_flow(tr_batch, e, F_flows, config["n_flows_F"])

            loss = loss_fun(z, z_ldetJ, prior_z, e, e_ldetJ, prior_e)
            loss_acc += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                OrderedDict(
                    {
                        "loss": "%.4f" % loss.item(),
                        "loss_avg": "%.4f" % (loss_acc / (j + 1)),
                    }
                )
            )
        scheduler.step()

        if (i + 1) % 2 == 0:
            path = config["save_models_dir_backup"]
        else:
            path = config["save_models_dir"]
            print("model_saved")

        for key in F_flows:
            torch.save(F_flows[key].state_dict(), path + "F_" + key + ".pth")
        for key in G_flows:
            torch.save(G_flows[key].state_dict(), path + "G_" + key + ".pth")
        torch.save(optimizer.state_dict(), path + "optimizer.pth")
        torch.save(scheduler.state_dict(), path + "scheduler.pth")

        if not os.path.exists(os.path.join(os.path.pardir)):
            os.makedirs(os.path.join(config["losses"], os.path.pardir))

        with open(config["losses"], "a") as file:
            file.write(
                "Epoch: {}/{} Loss: {:.4f} Time: {}\n".format(
                    i + 1,
                    config["n_epochs"],
                    loss_acc / (len(dataloader_pointflow) + 1),
                    str(datetime.datetime.now().time()).split(".")[0],
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
