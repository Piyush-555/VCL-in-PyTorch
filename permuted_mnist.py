import torch

import vcl
import dataset
import coresets
from models import PermutedModel


num_tasks = 5
num_epochs = 10
single_head = True
batch_size = 256

dataloaders = dataset.get_permuted_dataloaders(num_tasks)
model = PermutedModel()
model.cuda()

# Vanilla VCL
coreset_size = 200
coreset_method = coresets.attach_random_coreset_permuted
vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
            model, coreset_method, coreset_size, beta=0.01)

# Random Coreset VCL
coreset_size = 200
coreset_method = coresets.attach_random_coreset_permuted
vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
            model, coreset_method, coreset_size, beta=0.01)
