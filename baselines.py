import numpy as np
from tqdm import tqdm
import torch
import torchvision
from torch import nn
from torch.optim import Adam, SGD
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(outputs, targets):
    return np.mean(outputs.argmax(dim=-1).cpu().numpy() == targets.cpu().numpy())


def train(model, num_epochs, dataloader, single_head, task_id):
    lr_start = 1e-2

    if single_head:
        offset = 0
        output_nodes = 10
    else:
        output_nodes = model.classifier.out_features
        offset = task_id * output_nodes

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = SGD(model.parameters(), lr=lr_start, momentum=0.9)

    model.train()
    for epoch in tqdm(range(num_epochs)):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            targets -= offset
            net_out = F.softmax(model(inputs, task_id), dim=-1)
            loss = criterion(net_out, targets)
            loss.backward()
            optimizer.step()


def predict(model, dataloader, single_head, task_id):
    if single_head:
        offset = 0
        output_nodes = 10
    else:
        output_nodes = model.classifier.out_features
        offset = task_id * output_nodes

    model.train()
    accs = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        targets -= offset
        with torch.no_grad():
            net_out = F.softmax(model(inputs, task_id), dim=-1)
        accs.append(calculate_accuracy(net_out, targets))
    
    return np.mean(accs)


def run_baseline(num_tasks, single_head, num_epochs, dataloaders, model):

    if not single_head:
        assert 10 // num_tasks == 10 / num_tasks

    all_accs = np.empty(shape=(num_tasks, num_tasks))
    all_accs.fill(np.nan)
    for task_id in range(num_tasks):
        print("Starting Task", task_id + 1)
        if single_head:
            offset = 0
        else:
            offset = task_id * 10 // num_tasks

        # Train on non-coreset data
        trainloader, testloader = dataloaders[task_id]
        train(model, num_epochs, trainloader, single_head, task_id)
        print("Done Training Task", task_id + 1)

        # Evaluate on old tasks
        for task in range(task_id + 1):
            _, testloader_i = dataloaders[task]
            accuracy = predict(model, testloader_i, single_head, task)
            print("Task {} Accuracy: {}".format(task + 1, accuracy))
            all_accs[task_id][task] = accuracy
            # import pdb; pdb.set_trace()

    print(all_accs)
    return all_accs
