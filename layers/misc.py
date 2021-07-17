import torch
from torch import nn


class FlattenLayer(nn.Module):
    
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


def KL_DIV(mu_p, sig_p, mu_q, sig_q):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl
