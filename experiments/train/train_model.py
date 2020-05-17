import argparse
import datetime
import yaml
import os
import tqdm
from collections import OrderedDict

import torch
from torch import distributions
from torch.utils.data import DataLoader
import numpy as np

from data.datasets_pointflow import (
    CIFDatasetDecorator,
    ShapeNet15kPointClouds,
    CIFDatasetDecoratorMultiObject,
)
from utils.losses import loss_fun

# try:
#     from utils.MDS import multiDS
# except:
#     print("MDS failed to load")
from utils.MDS import multiDS

from models.models import model_init, model_load
from models.flows import F_flow_new, G_flow_new, F_flow, G_flow
from models.pointnet import Encoder
from experiments.test.metrics_eval import metrics_eval

from torch.utils.tensorboard import SummaryWriter


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_z = distributions.MultivariateNormal(
        torch.zeros(3), torch.eye(3)
    )  # * config['prior_z_var'])
    prior_e = distributions.MultivariateNormal(
        torch.zeros(config["emb_dim"]), torch.eye(config["emb_dim"])
    )  # * config['prior_e_var'])

    # each dataset needs to be decorated
    if config["use_random_dataloader"]:
        tr_sample_size = config["tr_sample_size"]
        te_sample_size = config["te_sample_size"]
        batch_size = config["batch_size_if_random_split"]
    else:
        tr_sample_size = config["tr_sample_size"]
        te_sample_size = config["te_sample_size"]
        batch_size = config["batch_size"]

    cloud_pointflow = ShapeNet15kPointClouds(
        tr_sample_size=tr_sample_size,
        te_sample_size=te_sample_size,
        root_dir=config["root_dir"],
        root_embs_dir=config["root_embs_dir"],
        normalize_per_shape=config["normalize_per_shape"],
        normalize_std_per_axis=config["normalize_std_per_axis"],
        split="train",
        scale=config["scale"],
        categories=config["categories"],
        random_subsample=True,
    )

    if config["use_random_dataloader"]:
        # cloud_pointflow = CIFDatasetDecorator(cloud_pointflow)
        cloud_pointflow = CIFDatasetDecoratorMultiObject(
            cloud_pointflow, config["num_of_points_per_object"]
        )

    if not os.path.exists(config["save_models_dir"]):
        os.makedirs(config["save_models_dir"])
    if not os.path.exists(config["save_models_dir_backup"]):
        os.makedirs(config["save_models_dir_backup"])

    dataloader_pointflow = DataLoader(
        cloud_pointflow, batch_size=batch_size, shuffle=True
    )

    # np.save(
    #     os.path.join(config["save_models_dir"], "train_set_mean.npy"),
    #     cloud_pointflow.all_points.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3),
    # )
    # np.save(
    #     os.path.join(config["save_models_dir"], "train_set_std.npy"),
    #     cloud_pointflow.all_points.reshape(-1, 3).std(axis=0).reshape(1, 1, 3),
    # )
    np.save(
        os.path.join(config["save_models_dir"], "train_set_mean.npy"),
        cloud_pointflow.all_points_mean,
    )
    np.save(
        os.path.join(config["save_models_dir"], "train_set_std.npy"),
        cloud_pointflow.all_points_std,
    )
    np.save(
        os.path.join(config["save_models_dir"], "train_set_idx.npy"),
        np.array(cloud_pointflow.shuffle_idx),
    )
    np.save(
        os.path.join(config["save_models_dir"], "val_set_mean.npy"),
        cloud_pointflow.all_points_mean,
    )
    np.save(
        os.path.join(config["save_models_dir"], "val_set_std.npy"),
        cloud_pointflow.all_points_std,
    )
    np.save(
        os.path.join(config["save_models_dir"], "val_set_idx.npy"),
        np.array(cloud_pointflow.shuffle_idx),
    )

    print("Preparing model for " + config["categories"][0])

    if config["load_models"]:
        F_flows, G_flows, optimizer, scheduler = model_load(config, device)
    else:
        F_flows, G_flows, optimizer, scheduler = model_init(config, device)

    # if config['current_lrate_mul']:
    #     optimizer.param_groups[0]['lr'] *= config['current_lrate_mul']

    if config["train_F"]:
        for key in F_flows:
            F_flows[key].train()
    else:
        for key in F_flows:
            F_flows[key].eval()

    if config["train_G"]:
        for key in G_flows:
            G_flows[key].train()
    else:
        for key in G_flows:
            G_flows[key].eval()

    print("Starting training...")

    train_writer = SummaryWriter(config["tensorboard_dir"] + "train")
    valid_writer = SummaryWriter(config["tensorboard_dir"] + "valid")

    global_step = 0

    pointnet = Encoder(
        load_pretrained=config["load_pretrained"],
        pretrained_path=config["pretrained_path"],
        zdim=32,
    ).to(device)

    for i in range(config["n_epochs"]):
        print("Epoch: {} / {}".format(i + 1, config["n_epochs"]))

        loss_acc_z = 0
        loss_acc_e = 0
        pbar = tqdm.tqdm(dataloader_pointflow, desc="Batch")

        optimizer.zero_grad()
        for j, datum in enumerate(pbar):
            # "idx_batch" -> indices of instances of a class
            idx_batch, embs_tr_batch = (datum["idx"], datum["train_points"])
            tr_batch = datum["points_to_decode"]

            # gradient was applied in previous step, so we can reset it now
            if (j - 1) > 0 and (j - 1) % config["aggregation_steps"] == 0:
                optimizer.zero_grad()

            if not config["use_random_dataloader"]:
                b, n_points, coords_num = tr_batch.shape
                idx_batch = (
                    idx_batch.unsqueeze(dim=-1)
                    .repeat(repeats=(1, n_points))
                    .reshape((-1,))
                )

            num_points_per_object = tr_batch.shape[1]
            tr_batch = (
                (
                    tr_batch.float()
                    + config["x_noise"] * torch.rand(tr_batch.shape)
                )
                .to(device)
                .reshape((-1, 3))
            )

            w_iter = pointnet(embs_tr_batch.to(device))
            w_iter = (
                w_iter.unsqueeze(dim=1)
                .expand(
                    [w_iter.shape[0], num_points_per_object, w_iter.shape[-1]]
                )
                .reshape((-1, w_iter.shape[-1]))
            )

            if config["use_new_g"]:
                e, e_ldetJ = G_flow_new(w_iter, G_flows, config["n_flows_G"])
            else:
                e, e_ldetJ = G_flow(
                    w_iter, G_flows, config["n_flows_G"], config["emb_dim"]
                )

            if config["use_new_f"]:
                z, z_ldetJ = F_flow_new(
                    tr_batch, e, F_flows, config["n_flows_F"]
                )
            else:
                z, z_ldetJ = F_flow(tr_batch, e, F_flows, config["n_flows_F"])

            loss_z, loss_e = loss_fun(
                z,
                z_ldetJ,
                prior_z,
                e,
                e_ldetJ,
                prior_e,
                _lambda=config["e_loss_scale"],
            )
            loss = loss_e + loss_z
            loss_acc_z += loss_z.item()
            loss_acc_e += loss_e.item()

            loss.backward()

            if j > 0 and j % config["aggregation_steps"] == 0:
                optimizer.step()

            train_writer.add_scalar("loss_z", loss_z, global_step=global_step)
            train_writer.add_scalar("loss_e", loss_e, global_step=global_step)

            pbar.set_postfix(
                OrderedDict(
                    {
                        "loss_z_avg": "%.4f" % (loss_acc_z / (j + 1)),
                        "loss_e_avg": "%.4f" % (loss_acc_e / (j + 1)),
                        "loss_avg": "%.4f"
                        % ((loss_acc_z + loss_acc_e) / (j + 1)),
                    }
                )
            )

            global_step += 1

        scheduler.step()

        if (i + 1) % 2 == 0:
            path = config["save_models_dir_backup"]
        else:
            path = config["save_models_dir"]

        try:
            for key in F_flows:
                torch.save(
                    F_flows[key].module.state_dict(),
                    path + "F_" + key + ".pth",
                )
            for key in G_flows:
                torch.save(
                    G_flows[key].module.state_dict(),
                    path + "G_" + key + ".pth",
                )
        except:
            for key in F_flows:
                torch.save(
                    F_flows[key].state_dict(), path + "F_" + key + ".pth"
                )
            for key in G_flows:
                torch.save(
                    G_flows[key].state_dict(), path + "G_" + key + ".pth"
                )

        torch.save(optimizer.state_dict(), path + "optimizer.pth")
        torch.save(scheduler.state_dict(), path + "scheduler.pth")
        torch.save(pointnet.state_dict(), path + "pointnet.pth")

        cov, mmd = metrics_eval(F_flows, config, device)

        with open(config["losses"], "a") as file:
            file.write(
                f"Epoch: {i + 1}/{config['n_epochs']} \t"
                + f"Loss_z: {loss_acc_z / (j + 1):.4f} \t"
                + f"Loss_e: {loss_acc_e / (j + 1):.4f} \t"
                + f"Total loss: {(loss_acc_z + loss_acc_e) / (j + 1):.4f} \t"
                + f"Coverage: {cov :.4f}% \t"
                + f"MMD: {mmd :.8f} \t"
                + f"Time: {str(datetime.datetime.now().time()).split('.')[0]}\n"
            )

        valid_writer.add_scalar(
            "loss_z", loss_acc_z / (j + 1), global_step=global_step
        )
        valid_writer.add_scalar(
            "loss_e", loss_acc_e / (j + 1), global_step=global_step
        )
        valid_writer.add_scalar("cov", cov, global_step=global_step)
        valid_writer.add_scalar("mmd", mmd, global_step=global_step)

        train_writer.close()
        valid_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
