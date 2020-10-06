import torch


def F_flow(x, e, F_flows, n_flows_F):
    ldetJ = 0.
    for n in range(n_flows_F):
        for k in range(3):
            h1, h2 = x[:, :2], x[:, 2, None]
            M = F_flows['MNet' + str(n) + '_' + str(k)](h1, e)
            A = F_flows['ANet' + str(n) + '_' + str(k)](h1, e)
            h2 = h2 * torch.exp(M) + A
            ldetJ += torch.sum(M, dim=1).view(-1, 1)
            x = torch.cat([h2, h1], dim=1)
    return x, ldetJ


def F_inv_flow(z, e, F_flows, n_flows_F):
    for n in range(n_flows_F-1, -1, -1):
        for k in range(2, -1, -1):
            h1, h2 = z[:, 1:], z[:, 0, None]
            M_inv = torch.exp(-F_flows['MNet' + str(n) + '_' + str(k)](h1, e))
            A = F_flows['ANet' + str(n) + '_' + str(k)](h1, e)
            h2 = (h2 - A) * M_inv
            z = torch.cat([h1, h2], dim=1)
    return z


def G_flow(w, G_flows, n_flows_G, emb_dim):
    ldetJ = 0.
    for n in range(n_flows_G):
        for k in range(2):
            h1, h2 = w[:, :emb_dim//2], w[:, emb_dim//2:]
            M = G_flows['MNet' + str(n) + '_' + str(k)](h1)
            A = G_flows['ANet' + str(n) + '_' + str(k)](h1)
            h2 = h2 * torch.exp(M) + A
            ldetJ += torch.sum(M, dim=1).view(-1, 1)
            w = torch.cat([h2, h1], dim=1)
    return w, ldetJ