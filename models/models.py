import argparse
import torch
import torch.nn as nn
from utils.model_utils import init_weights, optim
from models.architecture import F_AddNet, F_MulNet, G_AddNet, G_MulNet


def model_init(config: argparse.Namespace, device):
    F_flows = {}
    for n in range(config['n_flows_F']):
        for i in range(3):
            F_flows['MNet' + str(n) + str(i)] = F_MulNet(config['emb_dim'], config['n_neurons']).to(device)
            F_flows['MNet' + str(n) + str(i)].apply(init_weights)
            F_flows['ANet' + str(n) + str(i)] = F_AddNet(config['emb_dim'], config['n_neurons']).to(device)
            F_flows['ANet' + str(n) + str(i)].apply(init_weights)

    G_flows = {}
    for n in range(config['n_flows_G']):
        for i in range(2):
            G_flows['MNet' + str(n) + str(i)] = G_MulNet(config['emb_dim'], config['n_neurons']).to(device)
            G_flows['MNet' + str(n) + str(i)].apply(init_weights)
            G_flows['ANet' + str(n) + str(i)] = G_AddNet(config['emb_dim'], config['n_neurons']).to(device)
            G_flows['ANet' + str(n) + str(i)].apply(init_weights)

    if torch.cuda.device_count() > 1:
        for key in F_flows:
            F_flows[key] = nn.DataParallel(F_flows[key])
        for key in G_flows:
            G_flows[key] = nn.DataParallel(G_flows[key])

    optimizer = optim(F_flows, G_flows, config['l_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)

    return F_flows, G_flows, optimizer, scheduler


def model_load(config: argparse.Namespace, device, train=True):
    path = config['load_models_dir']

    F_flows = {}
    for n in range(config['n_flows_F']):
        for i in range(3):
            F_flows['MNet' + str(n) + str(i)] = F_MulNet(config['emb_dim'], config['n_neurons']).to(device)
            F_flows['ANet' + str(n) + str(i)] = F_AddNet(config['emb_dim'], config['n_neurons']).to(device)

    G_flows = {}
    for n in range(config['n_flows_G']):
        for i in range(2):
            G_flows['MNet' + str(n) + str(i)] = G_MulNet(config['emb_dim'], config['n_neurons']).to(device)
            G_flows['ANet' + str(n) + str(i)] = G_AddNet(config['emb_dim'], config['n_neurons']).to(device)

    if torch.cuda.device_count() > 1:
        for key in F_flows:
            F_flows[key] = nn.DataParallel(F_flows[key])
        for key in G_flows:
            G_flows[key] = nn.DataParallel(G_flows[key])

    for key in F_flows:
        F_flows[key].load_state_dict(torch.load(path + r'F_' + key + r'.pth'))
    for key in G_flows:
        G_flows[key].load_state_dict(torch.load(path + r'G_' + key + r'.pth'))

    if train:
        optimizer = optim(F_flows, G_flows, config['l_rate'])
        optimizer.load_state_dict(torch.load(path + r'optimizer.pth'))

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        scheduler.load_state_dict(torch.load(path + r'scheduler.pth'))
    else:
        optimizer, scheduler = None, None

    w = torch.load(path + r'w.pth').to(device)

    return F_flows, G_flows, optimizer, scheduler, w
