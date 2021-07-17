import torch
import numpy as np
import matplotlib.pyplot as plt

import vcl
import baselines
import dataset
import coresets
from models import SplitModel, BaselineModelMNIST
from experiment_base import initiate_experiment


num_tasks = 5
coreset_method = coresets.attach_random_coreset_split

split_class_distribution = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]

joint_class_distribution = [
    [0, 1],
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
]

@initiate_experiment
def splitmnist_baselines(num_epochs=100, batch_size=15000):
    filename = 'splitmnist_baselines'
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7, 4), dpi=300)
    ax = plt.gca()

    # VCL
    dataloaders = dataset.get_split_dataloaders(split_class_distribution, batch_size)
    single_head = False
    model = SplitModel()
    model.cuda()
    csz = 200
    beta = 0.01
    print("\nExperimenting VCL")
    all_accs = vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
                           model, coreset_method, csz, beta)
    accs = np.nanmean(all_accs, axis=1)
    print("Average accuracy after each task:", accs)
    np.save('logs/{}-vcl.npy'.format(filename), all_accs)
    plt.plot(np.arange(len(accs))+1, accs, label='VCL', marker='o')

    # Normal
    dataloaders = dataset.get_split_dataloaders(split_class_distribution, batch_size)
    single_head = False
    model = BaselineModelMNIST(2)
    model.cuda()
    print("\nExperimenting Normal-SGD")
    all_accs = baselines.run_baseline(num_tasks, single_head, num_epochs, dataloaders, model)
    accs = np.nanmean(all_accs, axis=1)
    print("Average accuracy after each task:", accs)
    np.save('logs/{}-normal.npy'.format(filename), all_accs)
    plt.plot(np.arange(len(accs))+1, accs, label='Normal-SGD', marker='o')

    # Joint
    dataloaders = dataset.get_split_dataloaders(joint_class_distribution, batch_size)
    single_head = True
    print("\nExperimenting Joint")
    accs = []
    for task_id in range(num_tasks):
        model = BaselineModelMNIST((task_id + 1) * 2)
        model.cuda()
        trainloader, testloader = dataloaders[task_id]
        # Train
        mod_num_epochs = num_epochs + task_id * int(num_epochs * 0.2)
        baselines.train(model, num_epochs, trainloader, single_head, task_id)
        print("Done Training Task", task_id + 1)
        # Predict
        accuracy = baselines.predict(model, testloader, single_head, task_id)
        print("Accuracy: {}".format(accuracy))
        accs.append(accuracy)
    print("Average accuracy after each task:", accs)
    np.save('logs/{}-joint.npy'.format(filename), accs)
    plt.plot(np.arange(len(accs))+1, accs, label='Joint', marker='o')


    ax.set_xticks(list(range(1, len(accs)+1)))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.set_title('SplitMNIST')
    ax.legend(title='Method')

    fig.savefig("plots/{}.png".format(filename), bbox_inches='tight')
    plt.close()


if __name__=='__main__':
    num_epochs = 50
    batch_size = 256
    splitmnist_baselines(num_epochs=num_epochs, batch_size=batch_size)
