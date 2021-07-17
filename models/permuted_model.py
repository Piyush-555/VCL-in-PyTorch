import torch.nn as nn
import torch.nn.functional as F

from layers import BBBLinear
from layers import FlattenLayer


class PermutedModel(nn.Module):

    def __init__(self):
        """
        input: 1 x 28 x 28
        output: 1 classifiers with 10 nodes
        hidden: [100, 100]
        """
        super().__init__()
        self.fc1 = BBBLinear(784, 100)
        self.fc2 = BBBLinear(100, 100)
        self.classifier = BBBLinear(100, 10)

    def forward(self, x, task_id):
        out = x.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.classifier(out)

    def get_kl(self, task_id):
        kl = 0.0
        kl += self.fc1.kl_loss()
        kl += self.fc2.kl_loss()
        kl += self.classifier.kl_loss()
        return kl

    def update_prior(self):
        self.fc1.prior_W_mu = self.fc1.W_mu.data
        self.fc1.prior_W_sigma = self.fc1.W_sigma.data
        if self.fc1.use_bias:
            self.fc1.prior_bias_mu = self.fc1.bias_mu.data
            self.fc1.prior_bias_sigma = self.fc1.bias_sigma.data

        self.fc2.prior_W_mu = self.fc2.W_mu.data
        self.fc2.prior_W_sigma = self.fc2.W_sigma.data
        if self.fc2.use_bias:
            self.fc2.prior_bias_mu = self.fc2.bias_mu.data
            self.fc2.prior_bias_sigma = self.fc2.bias_sigma.data

        self.classifier.prior_W_mu = self.classifier.W_mu.data
        self.classifier.prior_W_sigma = self.classifier.W_sigma.data
        if self.classifier.use_bias:
            self.classifier.prior_bias_mu = self.classifier.bias_mu.data
            self.classifier.prior_bias_sigma = self.classifier.bias_sigma.data
