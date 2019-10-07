import torch
import torch.nn as nn
import torch.nn.functional as F


class F_MulNet(nn.Module):
    def __init__(self, emb_dim, n_neurons):
        super(F_MulNet, self).__init__()

        self.layer0 = self.layer1 = nn.Sequential(
            nn.Linear(emb_dim, n_neurons//2),
            nn.LeakyReLU(0.1)
        )

        self.layer1 = nn.Sequential(
            nn.Linear(2, n_neurons//2),
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

    def forward(self, x, y):
        y = self.layer0(y)
        x = self.layer1(x)
        x = self.layer2(torch.cat([x, y], dim=1))

        _x = x
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        _x = x
        x = self.layer5(x)
        x = self.layer6(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        _x = x
        x = self.layer7(x)
        x = self.layer8(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        _x = x
        x = self.layer9(x)
        x = self.layer10(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        x = self.layer11(x)

        return x


class F_AddNet(nn.Module):
    def __init__(self, emb_dim, n_neurons):
        super(F_AddNet, self).__init__()

        self.layer0 = self.layer1 = nn.Sequential(
            nn.Linear(emb_dim, n_neurons//2),
            nn.LeakyReLU(0.1)
        )

        self.layer1 = nn.Sequential(
            nn.Linear(2, n_neurons//2),
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

    def forward(self, x, y):
        y = self.layer0(y)
        x = self.layer1(x)
        x = self.layer2(torch.cat([x, y], dim=1))

        _x = x
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        _x = x
        x = self.layer5(x)
        x = self.layer6(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        _x = x
        x = self.layer7(x)
        x = self.layer8(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        _x = x
        x = self.layer9(x)
        x = self.layer10(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        x = self.layer11(x)

        return x


class G_MulNet(nn.Module):
    def __init__(self, emb_dim, n_neurons):
        super(G_MulNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(emb_dim // 2, n_neurons),
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
            nn.Linear(n_neurons, emb_dim // 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)

        _x = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        _x = x
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        _x = x
        x = self.layer6(x)
        x = self.layer7(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        x = self.layer8(x)

        return x


class G_AddNet(nn.Module):
    def __init__(self, emb_dim, n_neurons):
        super(G_AddNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(emb_dim // 2, n_neurons),
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
            nn.Linear(n_neurons, emb_dim // 2)
        )

    def forward(self, x):
        x = self.layer1(x)

        _x = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        _x = x
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        _x = x
        x = self.layer6(x)
        x = self.layer7(x)
        x = F.leaky_relu(x + _x, negative_slope=0.1)

        x = self.layer8(x)

        return x


class Embeddings4Recon(nn.Module):
    def __init__(self, n_classes, emb_dim):
        super(Embeddings4Recon, self).__init__()
        self.emb_dim = emb_dim
        self.embs = nn.Parameter(torch.randn(n_classes, emb_dim))

    def forward(self, targets):
        targets = targets.view(-1, 1)
        return self.embs[targets].view(-1, self.emb_dim)
