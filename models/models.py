import argparse
import torch
import os
import torch.nn as nn
from utils.model_utils import init_weights, optim
from models.architecture import F_AddNet, F_MulNet, G_AddNet, G_MulNet
from models.pointnet import Encoder


def model_init(config: argparse.Namespace, device):
    F_flows = {}
    if config['use_new_f']:
        for n in range(config['n_flows_F']):
            for i in range(3):
                for j in range(3):
                    F_flows['MNet' + str(n) + '_' + str(i) + '_' + str(j)] = F_MulNet(j, config['emb_dim'], config['n_neurons'], type_emb=config['type_emb'], arch_type=config['arch_type']).to(device)
                    F_flows['MNet' + str(n) + '_' + str(i) + '_' + str(j)].apply(init_weights)
                    F_flows['ANet' + str(n) + '_' + str(i) + '_' + str(j)] = F_AddNet(j, config['emb_dim'], config['n_neurons'], type_emb=config['type_emb'], arch_type=config['arch_type']).to(device)
                    F_flows['ANet' + str(n) + '_' + str(i) + '_' + str(j)].apply(init_weights)
    else:
        for n in range(config['n_flows_F']):
            for i in range(3):
                F_flows['MNet' + str(n) + '_' + str(i)] = F_MulNet(2, config['emb_dim'], config['n_neurons'], type_emb=config['type_emb'], arch_type=config['arch_type']).to(device)
                F_flows['MNet' + str(n) + '_' + str(i)].apply(init_weights)
                F_flows['ANet' + str(n) + '_' + str(i)] = F_AddNet(2, config['emb_dim'], config['n_neurons'], type_emb=config['type_emb'], arch_type=config['arch_type']).to(device)
                F_flows['ANet' + str(n) + '_' + str(i)].apply(init_weights)

    G_flows = {}
    if config['use_new_g']:
        for n in range(config['n_flows_G']):
            for i in range(2):
                for j in range(2):
                    G_flows['MNet' + str(n) + '_' + str(i) + '_' + str(j)] = G_MulNet(config['emb_dim'] // (2 ** (n + 1)), config['n_neurons'], arch_type=config['arch_type']).to(device)
                    G_flows['MNet' + str(n) + '_' + str(i) + '_' + str(j)].apply(init_weights)
                    G_flows['ANet' + str(n) + '_' + str(i) + '_' + str(j)] = G_AddNet(config['emb_dim'] // (2 ** (n + 1)), config['n_neurons'], arch_type=config['arch_type']).to(device)
                    G_flows['ANet' + str(n) + '_' + str(i) + '_' + str(j)].apply(init_weights)

    else:
        for n in range(config['n_flows_G']):
            for i in range(2):
                G_flows['MNet' + str(n) + '_' + str(i)] = G_MulNet(config['emb_dim'] // 2, config['n_neurons'], arch_type=config['arch_type']).to(device)
                G_flows['MNet' + str(n) + '_' + str(i)].apply(init_weights)
                G_flows['ANet' + str(n) + '_' + str(i)] = G_AddNet(config['emb_dim'] // 2, config['n_neurons'], arch_type=config['arch_type']).to(device)
                G_flows['ANet' + str(n) + '_' + str(i)].apply(init_weights)

    print('CUDA:', torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        for key in F_flows:
            F_flows[key] = nn.DataParallel(F_flows[key])
        for key in G_flows:
            G_flows[key] = nn.DataParallel(G_flows[key])

    optimizer = optim(F_flows, G_flows, config['l_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    pointnet = Encoder(
        load_pretrained=False,
        pretrained_path=None,
        zdim=config["emb_dim"],
    ).to(device)

    return F_flows, G_flows, pointnet, optimizer, scheduler


def model_load(config: argparse.Namespace, device, train=True):
    path = config['load_models_dir']

    F_flows = {}
    if config['use_new_f']:
        for n in range(config['n_flows_F']):
            for i in range(3):
                for j in range(3):
                    F_flows['MNet' + str(n) + '_' + str(i) + '_' + str(j)] = F_MulNet(j, config['emb_dim'], config['n_neurons'], type_emb=config['type_emb'], arch_type=config['arch_type']).to(device)
                    F_flows['ANet' + str(n) + '_' + str(i) + '_' + str(j)] = F_AddNet(j, config['emb_dim'], config['n_neurons'], type_emb=config['type_emb'], arch_type=config['arch_type']).to(device)
    else:
        for n in range(config['n_flows_F']):
            for i in range(3):
                F_flows['MNet' + str(n) + '_' + str(i)] = F_MulNet(2, config['emb_dim'], config['n_neurons'], type_emb=config['type_emb'], arch_type=config['arch_type']).to(device)
                F_flows['ANet' + str(n) + '_' + str(i)] = F_AddNet(2, config['emb_dim'], config['n_neurons'], type_emb=config['type_emb'], arch_type=config['arch_type']).to(device)

    G_flows = {}
    if config['use_new_g']:
        for n in range(config['n_flows_G']):
            for i in range(2):
                for j in range(2):
                    G_flows['MNet' + str(n) + '_' + str(i) + '_' + str(j)] = G_MulNet(config['emb_dim'] // (2 ** (n + 1)), config['n_neurons'], arch_type=config['arch_type']).to(device)
                    G_flows['ANet' + str(n) + '_' + str(i) + '_' + str(j)] = G_AddNet(config['emb_dim'] // (2 ** (n + 1)), config['n_neurons'], arch_type=config['arch_type']).to(device)

    else:
        for n in range(config['n_flows_G']):
            for i in range(2):
                G_flows['MNet' + str(n) + '_' + str(i)] = G_MulNet(config['emb_dim'] // 2, config['n_neurons'], arch_type=config['arch_type']).to(device)
                G_flows['ANet' + str(n) + '_' + str(i)] = G_AddNet(config['emb_dim'] // 2, config['n_neurons'], arch_type=config['arch_type']).to(device)

    for key in F_flows:
        F_flows[key].load_state_dict(torch.load(path + r'F_' + key + r'.pth'))
    for key in G_flows:
        G_flows[key].load_state_dict(torch.load(path + r'G_' + key + r'.pth'))

    if torch.cuda.device_count() > 1:
        for key in F_flows:
            F_flows[key] = nn.DataParallel(F_flows[key])
        for key in G_flows:
            G_flows[key] = nn.DataParallel(G_flows[key])

    if train:
        optimizer = optim(F_flows, G_flows, config['l_rate'])
        optimizer.load_state_dict(torch.load(path + r'optimizer.pth'))

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        scheduler.load_state_dict(torch.load(path + r'scheduler.pth'))
    else:
        optimizer, scheduler = None, None

    pointnet = Encoder(
        load_pretrained=False,
        pretrained_path=None,
        zdim=config["emb_dim"],
    ).to(device)

    pointnet.load_state_dict(
        torch.load(os.path.join(config["load_models_dir"], "pointnet.pth"))
    )

    return F_flows, G_flows, pointnet, optimizer, scheduler
