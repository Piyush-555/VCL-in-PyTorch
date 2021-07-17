import torch.nn as nn
import torch.nn.functional as F

from layers import BBBLinear, BBBConv2d
from layers import FlattenLayer


class SplitModel(nn.Module):

    def __init__(self):
        """
        input: 1 x 28 x 28
        output: 5 classifiers 2 nodes each
        hidden: [256, 256]
        """
        super().__init__()
        self.fc1 = BBBLinear(784, 256)
        self.fc2 = BBBLinear(256, 256)
        self.classifiers = nn.ModuleList([BBBLinear(256, 2) for i in range(5)])

    def forward(self, x, task_id):
        out = x.view(-1, 784)
        for layer in self.children():
            if layer.__class__ is not nn.ModuleList:
                out = F.relu(layer(out))
        return self.classifiers[task_id](out)

    def get_kl(self, task_id):
        kl = 0.0
        for layer in self.children():
            if layer.__class__ is not nn.ModuleList:
                kl += layer.kl_loss()
        kl += self.classifiers[task_id].kl_loss()
        return kl

    def update_prior(self):
        for layer in self.children():
            if layer.__class__ is not nn.ModuleList:
                layer.prior_W_mu = layer.W_mu.data
                layer.prior_W_sigma = layer.W_sigma.data
                if layer.use_bias:
                    layer.prior_bias_mu = layer.bias_mu.data
                    layer.prior_bias_sigma = layer.bias_sigma.data
