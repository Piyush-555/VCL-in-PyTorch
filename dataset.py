import numpy as np
import torch
import torchvision
from torchvision import transforms

################## SplitMNIST ##################
def _extract_class_specific_idx(dataset, target_classes):
    """
    dataset: torchvision.datasets.MNIST
    target_classes: list
    """
    idx = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for target in target_classes:
        idx = idx | (dataset.targets==target)
    
    return idx


def get_split_dataloaders(class_distribution, batch_size=256):
    """
    class_distribution: list[list]
    """
    rsz = 28
    transform = transforms.Compose([
        transforms.Resize((rsz, rsz)),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.MNIST
    trainset = dataset(root="./data", train=True, download=True, transform=transform)
    testset = dataset(root="./data", train=False, download=True, transform=transform)

    dataloaders = []

    for classes in class_distribution:
        train_idx = _extract_class_specific_idx(trainset, classes)
        train_idx = torch.where(train_idx)[0]
        sub_train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        sub_train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, sampler=sub_train_sampler)

        test_idx = _extract_class_specific_idx(testset, classes)
        test_idx = torch.where(test_idx)[0]
        sub_test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
        sub_test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=sub_test_sampler)
        
        dataloaders.append((sub_train_loader, sub_test_loader))

    return dataloaders


################## PermutedMNIST ##################
def get_permuted_dataloaders(num_tasks, batch_size=256):
    dataset = torchvision.datasets.MNIST
    dataloaders = []

    for task in range(num_tasks):
        if task == 0:
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor()
            ])
        else:
            rng_permute = np.random.RandomState(task)
            idx_permute = torch.from_numpy(rng_permute.permutation(784))
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x, idx=idx_permute: x.view(-1)[idx].view(1, 28, 28))
            ])

        sub_trainset = dataset(root="./data", train=True, download=True, transform=transform)
        sub_train_sampler = torch.utils.data.RandomSampler(sub_trainset)
        sub_train_loader = torch.utils.data.DataLoader(
            sub_trainset, batch_size=batch_size, sampler=sub_train_sampler)

        sub_testset = dataset(root="./data", train=False, download=True, transform=transform)
        sub_test_sampler = torch.utils.data.RandomSampler(sub_testset)
        sub_test_loader = torch.utils.data.DataLoader(
            sub_testset, batch_size=batch_size, sampler=sub_test_sampler)

        dataloaders.append((sub_train_loader, sub_test_loader))

    return dataloaders


def get_joint_permuted_dataloaders(num_tasks, batch_size=256):
    dataset = torchvision.datasets.MNIST
    dataloaders = []
    transforms_list = []

    for task in range(num_tasks):
        if task == 0:
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor()
            ])
        else:
            rng_permute = np.random.RandomState(task)
            idx_permute = torch.from_numpy(rng_permute.permutation(784))
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x, idx=idx_permute: x.view(-1)[idx].view(1, 28, 28))
            ])
        transforms_list.append(transform)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(
            [dataset(root="./data", train=True, download=True, transform=transforms_list[i]) for i in range(num_tasks)]
            ),
            batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(
            [dataset(root="./data", train=False, download=True, transform=transforms_list[i]) for i in range(num_tasks)]
            ),
            batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
