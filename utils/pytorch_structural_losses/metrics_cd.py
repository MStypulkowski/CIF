import torch


def chamfer_distance(sample_pcs, ref_pcs, batch_size=None):
    """Use this function to calculate CD in our experiments."""
    if sample_pcs.dim() == 2:
        sample_pcs = sample_pcs.unsqueeze(0)
    if ref_pcs.dim() == 2:
        ref_pcs = ref_pcs.unsqueeze(0)

    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, f'REF:{N_ref} SMP:{N_sample}'

    batch_size = min(batch_size or N_sample, 70)

    cd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

    return torch.cat(cd_lst)


def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    xx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    yy = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    xx = (xx.transpose(2, 1) + yy - 2 * zz)
    return xx.min(1)[0], xx.min(2)[0]