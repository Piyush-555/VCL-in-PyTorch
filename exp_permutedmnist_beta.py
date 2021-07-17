import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import vcl
import dataset
import coresets
from models import PermutedModel
from experiment_base import initiate_experiment


num_tasks = 5
single_head = True
coreset_method = coresets.attach_random_coreset_permuted


@initiate_experiment
def permutedmnist_fixed_memory_vary_beta(num_epochs=100, batch_size=256, coreset_size=200, beta=None):
    filename = 'permutedmnist_fixed_memory_vary_beta'
    assert beta.__class__ is list
    #print("Beta:", beta, "with mean KL Divergence")
    dataloaders = dataset.get_permuted_dataloaders(num_tasks)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7, 4), dpi=300)
    ax = plt.gca()

    for i in range(len(beta)):
        bt= beta[i]
        print("\nExperimenting Beta value:", bt)
        model = PermutedModel()
        model.cuda()
        if "{}-{}.npy".format(filename, bt) in os.listdir("logs/"):
            print("Loading existing checkpoint..")
            all_accs = np.load("logs/{}-{}.npy".format(filename, bt))
        else:
            all_accs = vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
                                   model, coreset_method, coreset_size, bt)
            np.save('logs/{}-{}.npy'.format(filename, bt), all_accs)
        accs = np.nanmean(all_accs, axis=1)
        print("Average accuracy after each task:", accs)
        plt.plot(np.arange(len(accs))+1, accs, label=str(beta[i]), marker='o')
    
    ax.set_xticks(list(range(1, len(accs)+1)))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.legend(title="beta")
    ax.set_title("VCL - PermutedMNIST vary beta with fixed memory")

    fig.savefig("plots/{}.png".format(filename), bbox_inches='tight')
    plt.close()
        

if __name__=='__main__':
    num_epochs = 100
    batch_size = 256
    beta = [1e-3,1e-2,1e-1,1,1e1,1e2]
    coreset_size = 200
    permutedmnist_fixed_memory_vary_beta(num_epochs=num_epochs, batch_size=batch_size, coreset_size=coreset_size, beta=beta)
