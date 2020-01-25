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


def F_flow_new(x, e, F_flows, n_flows_F):
    ldetJ = 0.
    for n in range(n_flows_F):
        for k in range(3):
            h1, h2, h3 = x[:, 0, None], x[:, 1, None], x[:, 2, None]

            M0 = F_flows['MNet' + str(n) + '_' + str(k) + '_0'](e)
            A0 = F_flows['ANet' + str(n) + '_' + str(k) + '_0'](e)
            M1 = F_flows['MNet' + str(n) + '_' + str(k) + '_1'](h1, e)
            A1 = F_flows['ANet' + str(n) + '_' + str(k) + '_1'](h1, e)
            M2 = F_flows['MNet' + str(n) + '_' + str(k) + '_2'](torch.cat([h1, h2], dim=1), e)
            A2 = F_flows['ANet' + str(n) + '_' + str(k) + '_2'](torch.cat([h1, h2], dim=1), e)

            h1 = h1 * torch.exp(M0) + A0
            h2 = h2 * torch.exp(M1) + A1
            h3 = h3 * torch.exp(M2) + A2

            ldetJ += torch.sum(M0 + M1 + M2, dim=1).view(-1, 1)

            x = torch.cat([h2, h3, h1], dim=1)
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


def F_inv_flow_new(z, e, F_flows, n_flows_F):
    for n in range(n_flows_F-1, -1, -1):
        for k in range(2, -1, -1):
            h1, h2, h3 = z[:, 2, None], z[:, 0, None], z[:, 1, None]

            M0 = F_flows['MNet' + str(n) + '_' + str(k) + '_0'](e)
            A0 = F_flows['ANet' + str(n) + '_' + str(k) + '_0'](e)
            M1 = F_flows['MNet' + str(n) + '_' + str(k) + '_1'](h1, e)
            A1 = F_flows['ANet' + str(n) + '_' + str(k) + '_1'](h1, e)
            M2 = F_flows['MNet' + str(n) + '_' + str(k) + '_2'](torch.cat([h1, h2], dim=1), e)
            A2 = F_flows['ANet' + str(n) + '_' + str(k) + '_2'](torch.cat([h1, h2], dim=1), e)

            h1 = (h1 - A0) * torch.exp(-M0)
            h2 = (h2 - A1) * torch.exp(-M1)
            h3 = (h3 - A2) * torch.exp(-M2)

            z = torch.cat([h1, h2, h3], dim=1)
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


def G_flow_new(w, G_flows, n_flows_G):
    ldetJ = 0.
    e = torch.tensor([]).to(w.device)
    for n in range(n_flows_G):
        for k in range(2):
            for i in range(2):
                # split dimensions
                h1, h2 = w[:, :w.shape[1] // 2], w[:, w.shape[1] // 2:]

                # perform coupling transformation
                M = G_flows['MNet' + str(n) + '_' + str(k) + '_' + str(i)](h1)
                A = G_flows['ANet' + str(n) + '_' + str(k) + '_' + str(i)](h1)
                h2 = h2 * torch.exp(M) + A

                # calculate lod det of Jacobian
                ldetJ += torch.sum(M, dim=1).view(-1, 1)

                # concatenate with permutation
                w = torch.cat([h2, h1], dim=1)

        # factor out half of dimensions
        e = torch.cat([e, h1], dim=1)
        w = h2

    e = torch.cat([e, w], dim=1)
    return e, ldetJ
