import torch

import vcl
import dataset
import coresets
from models import SplitModel


"""
SGD w/ momentum 1e-2, 0.9

(faster) vcl = 96.450, coreset(200) = 97.062
epochs = 10
bsz = 256

(better) vcl = 97.542, coreset(200) = 98.342
epochs = 100
bsz = whole_dataset 
"""

num_tasks = 5
num_epochs = 10
single_head = False
batch_size = 256

class_distribution = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]
dataloaders = dataset.get_split_dataloaders(
    class_distribution, batch_size)
model = SplitModel()
model.cuda()

# Vanilla VCL
coreset_size = 0
coreset_method = coresets.attach_random_coreset_split
vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
            model, coreset_method, coreset_size, beta=0.01)

# Random Coreset VCL
coreset_size = 200
coreset_method = coresets.attach_random_coreset_split
vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
            model, coreset_method, coreset_size, beta=0.01)
