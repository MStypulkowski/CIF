import torch


def loss_fun(z, z_ldetJ, prior_z, e, e_ldetJ, prior_e):
    ll_z = prior_z.log_prob(z.cpu()).to(z.device) + z_ldetJ
    ll_e = prior_e.log_prob(e.cpu()).to(e.device) + e_ldetJ
    return -torch.mean(ll_z), -torch.mean(ll_e)


def loss_fun_ret(z, z_ldetJ, prior_z):
    ll_z = prior_z.log_prob(z.cpu()).to(z.device) + z_ldetJ
    return -torch.mean(ll_z)
