import argparse
import torch
import yaml
import tqdm
import numpy as np
from utils.metrics import MMD, coverage
from models.models import model_load
from models.flows import F_inv_flow_new, F_inv_flow, G_flow
from data.datasets_pointflow import CIFDatasetDecorator, ShapeNet15kPointClouds
from sklearn.mixture import GaussianMixture


def metrics_eval(F_flows, config, device):
    if config['use_random_dataloader']:
        tr_sample_size = 1
        te_sample_size = 1
    else:
        tr_sample_size = config['tr_sample_size']
        te_sample_size = config['te_sample_size']

    test_cloud = ShapeNet15kPointClouds(
        tr_sample_size=tr_sample_size,
        te_sample_size=te_sample_size,
        root_dir=config["root_dir"],
        root_embs_dir=config["root_embs_dir"],
        normalize_per_shape=config["normalize_per_shape"],
        normalize_std_per_axis=config["normalize_std_per_axis"],
        split="val",
        scale=config["scale"],
        categories=config["categories"],
        random_subsample=True,
    )

    # if config['use_random_dataloader']:
    #     test_cloud = CIFDatasetDecorator(test_cloud)
    #
    if (
            config["resume_dataset_mean"] is not None
            and config["resume_dataset_std"] is not None
    ):
        mean = np.load(config["resume_dataset_mean"])
        # mean = test_cloud.all_points_mean
        std = np.load(config["resume_dataset_std"])
        # std = test_cloud.all_points_std
        # print(mean.shape, std.shape)
        test_cloud.renormalize(mean, std)

        mean = torch.from_numpy(mean).to(device)
        std = torch.from_numpy(std).to(device)

    # mean = test_cloud.all_points_mean
    # std = test_cloud.all_points_std
    # ref_samples = torch.from_numpy(test_cloud.all_points[:, :2048, :] * std + mean).float()
    #
    # global_mean = torch.from_numpy(np.load(config["resume_dataset_mean"])).to(device)
    # global_std = torch.from_numpy(np.load(config["resume_dataset_std"])).to(device)

    for key in F_flows:
        F_flows[key].eval()

    #####
    # for key in G_flows:
    #     G_flows[key].eval()
    # with torch.no_grad():
    #     e, _ = G_flow(w, G_flows, config['n_flows_G'], config['emb_dim'])
    # stds = torch.std(e, dim=0)
    #####

    n_samples = test_cloud.all_points.shape[0]
    cloud_size = 2048

    # samples = torch.load(config['load_models_dir'] + 'metrics_samples.pth')
    covs_avg = []
    mmds_avg =[]
    # for stdev in ['GMM']:
    stdevs = config['prior_e_var'] if isinstance(config['prior_e_var'], list) else [config['prior_e_var']]

    for stdev in stdevs:
        covs = []
        mmds = []
        for i in range(5):
            ### GMM
            # for n in [i for i in range(10, 20)]:
            #     gmm = GaussianMixture(n)
            #     gmm.fit(e.cpu())
            #     weights_ = gmm.weights_
            #     means_ = gmm.means_
            #     covs_ = gmm.covariances_
            #
            #     embs4g = []
            #     for _ in range(n_samples):
            #         mix_id = np.random.choice(np.arange(n), p=weights_)
            #         embs4g.append(np.random.multivariate_normal(means_[mix_id], covs_[mix_id]))
            #     embs4g = torch.tensor(embs4g).float()
            #     print(weights_.shape, means_.shape, covs_.shape)
                ### end of GMM

                samples = []
                embs4g = torch.randn(n_samples, config['emb_dim']).to(device) * stdev #stds

                for sample_index in tqdm.trange(n_samples, desc="Sample"):
                    z = torch.randn(cloud_size, 3).to(device).float()
                    with torch.no_grad():
                        targets = torch.LongTensor(cloud_size, 1).fill_(sample_index)
                        embeddings4g = embs4g[targets].view(-1, config['emb_dim'])

                        if config['use_new_f']:
                            z = F_inv_flow_new(z, embeddings4g, F_flows, config['n_flows_F'])
                        else:
                            z = F_inv_flow(z, embeddings4g, F_flows, config['n_flows_F'])
                        # z = z * global_std + global_mean
                        z = z * std + mean
                        samples.append(z)

                samples = (
                    torch.cat(samples, dim=0)
                        .reshape((n_samples, cloud_size, 3))
                        .to(device)
                )
                torch.save(samples, config['load_models_dir'] + 'metrics_samples' + str(stdev).replace('.', '') + '_' + str(i) + '.pth')
                ref_samples = torch.from_numpy(test_cloud.all_points[:, :2048, :]).float().to(device)
                # print(f'ref samples device: {ref_samples.device}')
                ref_samples = ref_samples * std + mean
                # ref_samples = ref_samples.to(device)
                print(ref_samples.shape)

                if config["use_EMD"]:
                    covs.append(coverage(samples, ref_samples) * 100)
                    mmds.append(MMD(samples, ref_samples).item())

                else:
                    covs.append(coverage(samples, ref_samples, use_EMD=False) * 100)
                    mmds.append(MMD(samples, ref_samples, use_EMD=False).item())
                print('STD: ' + str(stdev))
                print('COV: ', covs)
                print('MMD: ', mmds)
        covs_avg.append(np.mean(covs))
        mmds_avg.append(np.mean(mmds))
    return covs_avg, mmds_avg


def main(config: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_flows = model_load(config, device, train=False)[0]
    # print(f"f_flows device: {next(F_flows['MNet0_0'].parameters()).device}")
    covs, mmds = metrics_eval(F_flows, config, device)
    with open(config['metrics_dir'], 'a') as file:
        # file.write('NEWF' + str(config['use_new_f']) + '_NEWG' + str(config['use_new_g']) +
        #            '_NF' + str(config['n_flows_F']) + '_NG' + str(config['n_flows_G']) +
        #            '_XN' + str(config['x_noise']) + '_WN' + str(config['w_noise']) +
        #            '_PE' + str(config['prior_e_var']) + '\n')
        for (cov, mmd, std) in zip(covs, mmds, config['prior_e_var']):
            print(f'Metrics for std {std}:')
            if config["use_EMD"]:
                # file.write("Coverage (EMD): {:.8f}% \n".format(cov))
                print("Coverage (EMD): {:.8f}%".format(cov))
                # file.write("MMD (EMD): {:.8f} \n\n".format(mmd))
                print("MMD (EMD): {:.8f}".format(mmd))

            else:
                # file.write("Coverage (CD): {:.8f}% \n".format(cov))
                print("Coverage (CD): {:.8f}%".format(cov))
                # file.write("MMD (CD): {:.8f} \n\n".format(mmd))
                print("MMD (CD): {:.8f}".format(mmd))


if __name__ == '__main__':
    # print(f'is CUDA AVAILABLE {torch.cuda.is_available()}')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
    else:
        with open(args.config) as f:
            main(yaml.full_load(f))
