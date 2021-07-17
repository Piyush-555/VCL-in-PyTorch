import torch.nn as nn
import torch.nn.functional as F


class PermutedBaselineModel(nn.Module):
    
    def __init__(self, outputs):
        """
        input: 1 x 28 x 28
        output: 1 classifiers with 10 nodes
        hidden: [100, 100]
        """
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        self.classifier = nn.Linear(100, outputs)

    def forward(self, x, task_id):
        out = x.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.classifier(out)


class BaselineModelMNIST(nn.Module):
    
    def __init__(self, outputs):
        """
        input: 1 x 28 x 28
        output: 5 classifiers 2 nodes each
        hidden: [256, 256]
        """
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.classifier = nn.Linear(256, outputs)

    def forward(self, x, task_id):
        out = x.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.classifier(out)
