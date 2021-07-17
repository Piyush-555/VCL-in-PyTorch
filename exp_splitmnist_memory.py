import torch
import numpy as np
import matplotlib.pyplot as plt

import vcl
import dataset
import coresets
from models import SplitModel
from experiment_base import initiate_experiment


num_tasks = 5
single_head = False
coreset_method = coresets.attach_random_coreset_split

class_distribution = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]

@initiate_experiment
def splitmnist_fixed_beta_vary_memory(num_epochs=100, batch_size=15000, coreset_size=None, beta=1):
    filename = 'splitmnist_fixed_beta_vary_memory'
    assert coreset_size.__class__ is list
    print("Beta:", beta, "with mean KL Divergence")
    dataloaders = dataset.get_split_dataloaders(
        class_distribution, batch_size)
    model = SplitModel()
    model.cuda()
    print("Model Arch:\n", model)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7, 4), dpi=300)
    ax = plt.gca()

    for i in range(len(coreset_size)):
        model = SplitModel()
        model.cuda()
        csz = coreset_size[i]
        print("\nExperimenting Coreset Size:", csz)
        all_accs = vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
                               model, coreset_method, csz, beta)
        accs = np.nanmean(all_accs, axis=1)
        print("Average accuracy after each task:", accs)
        np.save('logs/{}-{}.npy'.format(filename, csz), all_accs)
        plt.plot(np.arange(len(accs))+1, accs, label=str(coreset_size[i]), marker='o')

    ax.set_xticks(list(range(1, len(accs)+1)))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.set_title('VCL - SplitMNIST vary memory with fixed beta')
    ax.legend(title='Coreset Size')

    fig.savefig("plots/{}.png".format(filename), bbox_inches='tight')
    plt.close()


@initiate_experiment
def splitmnist_constant_KL_vary_memory(num_epochs=100, batch_size=15000, coreset_size=None, beta=1):
    filename = 'splitmnist_constant_KL_vary_memory'
    assert coreset_size.__class__ is list
    print("Beta:", beta, "with mean KL Divergence", "Constant KL Divergence")
    dataloaders = dataset.get_split_dataloaders(
        class_distribution, batch_size)
    model = SplitModel()
    model.cuda()
    print("Model Arch:\n", model)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7, 4), dpi=300)
    ax = plt.gca()

    for i in range(len(coreset_size)):
        model = SplitModel()
        model.cuda()
        csz = coreset_size[i]
        print("\nExperimenting Coreset Size:", csz)
        all_accs = vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
                               model, coreset_method, csz, beta, update_prior=False)
        accs = np.nanmean(all_accs, axis=1)
        print("Average accuracy after each task:", accs)
        np.save('logs/{}-{}.npy'.format(filename, csz), all_accs)
        plt.plot(np.arange(len(accs))+1, accs, label=str(coreset_size[i]), marker='o')
    
    ax.set_xticks(list(range(1, len(accs)+1)))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.set_title('Normal Bayesian NN - SplitMNIST vary memory with constant KL Divergence')
    ax.legend(title='Coreset Size')

    fig.savefig("plots/{}.png".format(filename), bbox_inches='tight')
    plt.close()


if __name__=='__main__':
    num_epochs = 10
    batch_size = 256
    coreset_size = [0, 20, 50, 200, 500, 1000]
    splitmnist_fixed_beta_vary_memory(num_epochs=num_epochs, batch_size=batch_size, coreset_size=coreset_size, beta=1)
    splitmnist_constant_KL_vary_memory(num_epochs=num_epochs, batch_size=batch_size, coreset_size=coreset_size, beta=1)
