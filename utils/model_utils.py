import torch


def init_weights(Layer):
    name = Layer.__class__.__name__
    if name == 'Linear':
        torch.nn.init.normal_(Layer.weight, mean=0, std=0.02)
        if Layer.bias is not None:
            torch.nn.init.constant_(Layer.bias, 0)


def optim(F_flows, G_flows, l_rate):
    params = []
    for key in F_flows:
        params += list(F_flows[key].parameters())

    for key in G_flows:
        params += list(G_flows[key].parameters())

    optimizer = torch.optim.Adam(params, lr=l_rate)
    return optimizer
