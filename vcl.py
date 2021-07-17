import numpy as np
from tqdm import tqdm
import torch
import torchvision
from torch import nn
from torch.optim import Adam, SGD
import torch.nn.functional as F

from layers import BBBConv2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ELBO(nn.Module):

    def __init__(self, model, train_size, beta):
        super().__init__()
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.beta = beta
        self.train_size = train_size

    def forward(self, outputs, targets, kl):
        assert not targets.requires_grad
        # print(F.nll_loss(outputs, targets, reduction='mean'), self.beta * kl / self.num_params)
        return F.nll_loss(outputs, targets, reduction='mean') + self.beta * kl / self.num_params


def calculate_accuracy(outputs, targets):
    return np.mean(outputs.argmax(dim=-1).cpu().numpy() == targets.cpu().numpy())


def train(model, num_epochs, dataloader, single_head, task_id, beta, T=10, replay=False):
    beta = 0 if replay else beta
    lr_start = 1e-3

    if single_head:
        offset = 0
        output_nodes = 10
    else:
        output_nodes = model.classifiers[0].out_features
        offset = task_id * output_nodes

    train_size = len(dataloader.dataset) if single_head else dataloader.sampler.indices.shape[0]
    elbo = ELBO(model, train_size, beta)
    optimizer = SGD(model.parameters(), lr=lr_start, momentum=0.9)

    model.train()
    for epoch in tqdm(range(num_epochs)):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            targets -= offset
            outputs = torch.zeros(inputs.shape[0], output_nodes, T, device=device)

            for i in range(T):
                net_out = model(inputs, task_id)
                outputs[:, :, i] = F.log_softmax(net_out, dim=-1)

            log_output = torch.logsumexp(outputs, dim=-1) - np.log(T)
            kl = model.get_kl(task_id)
            loss = elbo(log_output, targets, kl)
            loss.backward()
            optimizer.step()


def predict(model, dataloader, single_head, task_id, T=10):
    if single_head:
        offset = 0
        output_nodes = 10
    else:
        output_nodes = model.classifiers[0].out_features
        offset = task_id * output_nodes

    model.train()
    accs = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        targets -= offset
        outputs = torch.zeros(inputs.shape[0], output_nodes, T, device=device)

        for i in range(T):
            with torch.no_grad():
                net_out = model(inputs, task_id)
            outputs[:, :, i] = F.log_softmax(net_out, dim=-1)

        log_output = torch.logsumexp(outputs, dim=-1) - np.log(T)
        accs.append(calculate_accuracy(log_output, targets))
    
    return np.mean(accs)


def run_vcl(num_tasks, single_head, num_epochs, dataloaders, model,
            coreset_method, coreset_size=0, beta=1, update_prior=True):

    if not single_head:
        assert 10 // num_tasks == 10 / num_tasks
    
    coreset_list = []
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
        train(model, num_epochs, trainloader, single_head, task_id, beta)
        print("Done Training Task", task_id + 1)

        # Attach a new coreset
        if coreset_size > 0:
            coreset_method(coreset_list, trainloader, num_samples=coreset_size)

            # Replay old tasks using coresets
            for task in range(task_id + 1):
                print("Replaying Task", task + 1)
                train(model, num_epochs, coreset_list[task], single_head, task, beta, replay=True)
        print()

        # Evaluate on old tasks
        for task in range(task_id + 1):
            _, testloader_i = dataloaders[task]
            accuracy = predict(model, testloader_i, single_head, task)
            print("Task {} Accuracy: {}".format(task + 1, accuracy))
            all_accs[task_id][task] = accuracy
        print()
        if update_prior:
            model.update_prior()
    print(all_accs)
    return all_accs
