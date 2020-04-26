import argparse
import torch
import yaml
from models.flows import G_flow_new, G_flow
from models.models import model_load
from scipy.stats import normaltest


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, G_flows, _, _, w = model_load(config, device, train=False)

    for key in G_flows:
        G_flows[key].eval()

    with torch.no_grad():
        if config['use_new_g']:
            e, _ = G_flow_new(w, G_flows, config['n_flows_G'])
        else:
            e, _ = G_flow(w, G_flows, config['n_flows_G'], config['emb_dim'])
    print(e[:, 0].shape)
    print(torch.mean(e, dim=0).shape)
    means, stds = torch.mean(e, dim=0), torch.std(e, dim=0)

    for i, (mean, std) in enumerate(zip(means, stds)):
        print(f'Dim {i}: mean: {mean:.2f} std: {std:.2f}')
        _, p_val = normaltest(e[:, i].cpu())
        if p_val >= 0.05:
            print('True')

    # with open('normality_test.txt', 'a') as file:
    #     file.write('Mean of means: {:.4f} std of means: {:.4f}'.format(torch.mean(means).item(), torch.std(means).item()))
    #     file.write('Mean of stds: {:.4f} std of stds: {:.4f}'.format(torch.mean(stds).item(), torch.std(stds).item()))
    print('Mean of means: {:.4f} std of means: {:.4f}'.format(torch.mean(means).item(), torch.std(means).item()))
    print('Mean of stds: {:.4f} std of stds: {:.4f}'.format(torch.mean(stds).item(), torch.std(stds).item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
