import torch
import torch.nn as nn
import torch.nn.functional as F


class F_MulNet(nn.Module):
    def __init__(self, in_dim, emb_dim, n_neurons, type_emb='1l', arch_type='res_net'):
        super(F_MulNet, self).__init__()
        # print(in_dim, emb_dim, n_neurons, type_emb, arch_type)
        self.in_dim = in_dim
        self.type_emb = type_emb
        self.arch_type = arch_type

        if in_dim == 0:
            self.layer1 = nn.Sequential(
                nn.Linear(emb_dim, n_neurons),
                nn.LeakyReLU(0.1)
            )

        else:
            if type_emb == '1l':
                self.layer0 = nn.Sequential(
                    nn.Linear(emb_dim, n_neurons//2),
                    nn.LeakyReLU(0.1)
                )
            elif type_emb == '5l':
                self.layer0 = nn.Sequential(
                    nn.Linear(emb_dim, n_neurons//2),
                    nn.LeakyReLU(0.1),
                    nn.Linear(n_neurons // 2, n_neurons // 2),
                    nn.LeakyReLU(0.1),
                    nn.Linear(n_neurons // 2, n_neurons // 2),
                    nn.LeakyReLU(0.1),
                    nn.Linear(n_neurons // 2, n_neurons // 2),
                    nn.LeakyReLU(0.1),
                    nn.Linear(n_neurons // 2, n_neurons // 2),
                    nn.LeakyReLU(0.1)
                )
            elif type_emb == '2r':
                self.layer00 = nn.Sequential(
                    nn.Linear(emb_dim, n_neurons//2),
                    nn.LeakyReLU(0.1)
                )
                self.layer01 = nn.Sequential(
                    nn.Linear(n_neurons//2, n_neurons//2),
                    nn.LeakyReLU(0.1)
                )
                self.layer02 = nn.Linear(n_neurons//2, n_neurons//2)

                self.layer03 = nn.Sequential(
                    nn.Linear(n_neurons // 2, n_neurons // 2),
                    nn.LeakyReLU(0.1)
                )
                self.layer04 = nn.Linear(n_neurons // 2, n_neurons // 2)

            else:
                raise ValueError('emb layer type not valid')

            self.layer1 = nn.Sequential(
                nn.Linear(in_dim, n_neurons // 2),
                nn.LeakyReLU(0.1)
            )

        self.layer2 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer4 = nn.Linear(n_neurons, n_neurons)

        self.layer5 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer6 = nn.Linear(n_neurons, n_neurons)

        self.layer7 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer8 = nn.Linear(n_neurons, n_neurons)

        self.layer9 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer10 = nn.Linear(n_neurons, n_neurons)

        self.layer11 = nn.Sequential(
            nn.Linear(n_neurons, 1),
            nn.Tanh()
        )

    def forward(self, x, y=None):
        if self.in_dim == 0:
            x = self.layer1(x)
            x = self.layer2(x)

        else:
            if self.type_emb == '2r':
                y = self.layer00(y)

                _y = y
                y = self.layer01(y)
                y = self.layer02(y)
                y = F.leaky_relu(y + _y, negative_slope=0.1)

                _y = y
                y = self.layer03(y)
                y = self.layer04(y)
                y = F.leaky_relu(y + _y, negative_slope=0.1)
            else:
                y = self.layer0(y)

            x = self.layer1(x)
            x = self.layer2(torch.cat([x, y], dim=1))

        x1 = x
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.leaky_relu(x + x1, negative_slope=0.1)

        x2 = x
        x = self.layer5(x)
        x = self.layer6(x)

        if self.arch_type == 'dense_net':
            x = F.leaky_relu(x + x1 + x2, negative_slope=0.1)
        elif self.arch_type == 'res_net':
            x = F.leaky_relu(x + x2, negative_slope=0.1)

        x3 = x
        x = self.layer7(x)
        x = self.layer8(x)

        if self.arch_type == 'dense_net':
            x = F.leaky_relu(x + x1 + x2 + x3, negative_slope=0.1)
        elif self.arch_type == 'res_net':
            x = F.leaky_relu(x + x3, negative_slope=0.1)

        x4 = x
        x = self.layer9(x)
        x = self.layer10(x)

        if self.arch_type == 'dense_net':
            x = F.leaky_relu(x + x1 + x2 + x3 + x4, negative_slope=0.1)
        elif self.arch_type == 'res_net':
            x = F.leaky_relu(x + x4, negative_slope=0.1)

        x = self.layer11(x)

        return x


class F_AddNet(nn.Module):
    def __init__(self, in_dim, emb_dim, n_neurons, type_emb='1l', arch_type='res_net'):
        super(F_AddNet, self).__init__()
        # print(in_dim, emb_dim, n_neurons, type_emb, arch_type)
        self.in_dim = in_dim
        self.type_emb = type_emb
        self.arch_type = arch_type

        if in_dim == 0:
            self.layer1 = nn.Sequential(
                nn.Linear(emb_dim, n_neurons),
                nn.LeakyReLU(0.1)
            )

        else:
            if type_emb == '1l':
                self.layer0 = nn.Sequential(
                    nn.Linear(emb_dim, n_neurons // 2),
                    nn.LeakyReLU(0.1)
                )
            elif type_emb == '5l':
                self.layer0 = nn.Sequential(
                    nn.Linear(emb_dim, n_neurons // 2),
                    nn.LeakyReLU(0.1),
                    nn.Linear(n_neurons // 2, n_neurons // 2),
                    nn.LeakyReLU(0.1),
                    nn.Linear(n_neurons // 2, n_neurons // 2),
                    nn.LeakyReLU(0.1),
                    nn.Linear(n_neurons // 2, n_neurons // 2),
                    nn.LeakyReLU(0.1),
                    nn.Linear(n_neurons // 2, n_neurons // 2),
                    nn.LeakyReLU(0.1)
                )
            elif type_emb == '2r':
                self.layer00 = nn.Sequential(
                    nn.Linear(emb_dim, n_neurons // 2),
                    nn.LeakyReLU(0.1)
                )
                self.layer01 = nn.Sequential(
                    nn.Linear(n_neurons // 2, n_neurons // 2),
                    nn.LeakyReLU(0.1)
                )
                self.layer02 = nn.Linear(n_neurons // 2, n_neurons // 2)

                self.layer03 = nn.Sequential(
                    nn.Linear(n_neurons // 2, n_neurons // 2),
                    nn.LeakyReLU(0.1)
                )
                self.layer04 = nn.Linear(n_neurons // 2, n_neurons // 2)

            else:
                raise ValueError('emb layer type not valid')

            self.layer1 = nn.Sequential(
                nn.Linear(in_dim, n_neurons // 2),
                nn.LeakyReLU(0.1)
            )

        self.layer2 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer4 = nn.Linear(n_neurons, n_neurons)

        self.layer5 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer6 = nn.Linear(n_neurons, n_neurons)

        self.layer7 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer8 = nn.Linear(n_neurons, n_neurons)

        self.layer9 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer10 = nn.Linear(n_neurons, n_neurons)

        self.layer11 = nn.Sequential(
            nn.Linear(n_neurons, 1)
        )

    def forward(self, x, y=None):
        if self.in_dim == 0:
            x = self.layer1(x)
            x = self.layer2(x)

        else:
            if self.type_emb == '2r':
                y = self.layer00(y)

                _y = y
                y = self.layer01(y)
                y = self.layer02(y)
                y = F.leaky_relu(y + _y, negative_slope=0.1)

                _y = y
                y = self.layer03(y)
                y = self.layer04(y)
                y = F.leaky_relu(y + _y, negative_slope=0.1)
            else:
                y = self.layer0(y)

            x = self.layer1(x)
            x = self.layer2(torch.cat([x, y], dim=1))

        x1 = x
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.leaky_relu(x + x1, negative_slope=0.1)

        if self.arch_type == 'dense_net':
            x2 = x
        elif self.arch_type == 'res_net':
            x2 = x

        x = self.layer5(x)
        x = self.layer6(x)
        x = F.leaky_relu(x + x1 + x2, negative_slope=0.1)

        if self.arch_type == 'dense_net':
            x3 = x
        elif self.arch_type == 'res_net':
            x3 = x

        x = self.layer7(x)
        x = self.layer8(x)
        x = F.leaky_relu(x + x1 + x2 + x3, negative_slope=0.1)

        if self.arch_type == 'dense_net':
            x4 = x
        elif self.arch_type == 'res_net':
            x4 = x

        x = self.layer9(x)
        x = self.layer10(x)
        x = F.leaky_relu(x + x1 + x2 + x3 + x4, negative_slope=0.1)

        x = self.layer11(x)

        return x


class G_MulNet(nn.Module):
    def __init__(self, in_dim, n_neurons, arch_type='res_net'):
        super(G_MulNet, self).__init__()
        # print(in_dim, n_neurons, arch_type)
        self.arch_type = arch_type

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer3 = nn.Linear(n_neurons, n_neurons)

        self.layer4 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer5 = nn.Linear(n_neurons, n_neurons)

        self.layer6 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer7 = nn.Linear(n_neurons, n_neurons)

        self.layer8 = nn.Sequential(
            nn.Linear(n_neurons, in_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)

        x1 = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.leaky_relu(x + x1, negative_slope=0.1)

        if self.arch_type == 'dense_net':
            x2 = x
        elif self.arch_type == 'res_net':
            x2 = x

        x = self.layer4(x)
        x = self.layer5(x)
        x = F.leaky_relu(x + x1 + x2, negative_slope=0.1)

        if self.arch_type == 'dense_net':
            x3 = x
        elif self.arch_type == 'res_net':
            x3 = x

        x = self.layer6(x)
        x = self.layer7(x)
        x = F.leaky_relu(x + x1 + x2 + x3, negative_slope=0.1)

        x = self.layer8(x)

        return x


class G_AddNet(nn.Module):
    def __init__(self, in_dim, n_neurons, arch_type='res_net'):
        super(G_AddNet, self).__init__()
        # print(in_dim, n_neurons, arch_type)
        self.arch_type = arch_type

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer3 = nn.Linear(n_neurons, n_neurons)

        self.layer4 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer5 = nn.Linear(n_neurons, n_neurons)

        self.layer6 = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.LeakyReLU(0.1)
        )

        self.layer7 = nn.Linear(n_neurons, n_neurons)

        self.layer8 = nn.Sequential(
            nn.Linear(n_neurons, in_dim)
        )

    def forward(self, x):
        x = self.layer1(x)

        x1 = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.leaky_relu(x + x1, negative_slope=0.1)

        if self.arch_type == 'dense_net':
            x2 = x
        elif self.arch_type == 'res_net':
            x2 = x

        x = self.layer4(x)
        x = self.layer5(x)
        x = F.leaky_relu(x + x1 + x2, negative_slope=0.1)

        if self.arch_type == 'dense_net':
            x3 = x
        elif self.arch_type == 'res_net':
            x3 = x

        x = self.layer6(x)
        x = self.layer7(x)
        x = F.leaky_relu(x + x1 + x2 + x3, negative_slope=0.1)

        x = self.layer8(x)

        return x


class W4Recon(nn.Module):
    def __init__(self, config):
        super(W4Recon, self).__init__()
        self.emb_dim = config['emb_dim']
        self.embs = nn.Parameter(torch.load(config['load_models_dir'] + 'w.pth'))

    def forward(self):
        # targets = targets.view(-1, 1)
        # return self.embs[targets].view(-1, self.emb_dim)
        return self.embs.view(-1, self.emb_dim)
