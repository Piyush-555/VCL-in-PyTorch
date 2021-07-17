import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .misc import KL_DIV


class BBBLinear(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.01,
            }
        self.prior_W_mu = torch.tensor(priors['prior_mu'])
        self.prior_W_sigma = torch.tensor(priors['prior_sigma'])
        self.prior_bias_mu = torch.tensor(priors['prior_mu'])
        self.prior_bias_sigma = torch.tensor(priors['prior_sigma'])

        self.W_mu = Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(0, 0.1)
        self.W_rho.data.fill_(-3)

        if self.use_bias:
            self.bias_mu.data.normal_(0, 0.1)
            self.bias_rho.data.fill_(-3)

    def forward(self, x, sample=True):

        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, bias_var)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(act_mu.device)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = KL_DIV(self.prior_W_mu, self.prior_W_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_bias_mu, self.prior_bias_sigma, self.bias_mu, self.bias_sigma)
        return kl
