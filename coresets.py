import torch


def attach_random_coreset_split(coresets, sub_train_loader, num_samples=200):
    """
    coresets: list of collection of coreset dataloaders
    sub_train_loader: loader from which a random coreset is to be drawn
    num_samples: number of samples in each coreset
    """
    task_indices = sub_train_loader.sampler.indices
    shuffled_task_indices = task_indices[torch.randperm(len(task_indices))]
    coreset_indices = shuffled_task_indices[:num_samples]
    sub_train_loader.sampler.indices = shuffled_task_indices[num_samples:]  # Delete coreset from orginal data
    coreset_sampler = torch.utils.data.SubsetRandomSampler(coreset_indices)
    coreset_loader = torch.utils.data.DataLoader(
        sub_train_loader.dataset, batch_size=sub_train_loader.batch_size, sampler=coreset_sampler)
    coresets.append(coreset_loader)


def attach_random_coreset_permuted(coresets, sub_train_loader, num_samples=200):
    """
    coresets: list of collection of coreset dataloaders
    sub_train_loader: loader from which a random coreset is to be drawn
    num_samples: number of samples in each coreset
    """
    shuffled_task_indices = torch.randperm(len(sub_train_loader.dataset))
    coreset_indices = shuffled_task_indices[:num_samples]
    coreset_sampler = torch.utils.data.SubsetRandomSampler(coreset_indices)
    coreset_loader = torch.utils.data.DataLoader(
        sub_train_loader.dataset, batch_size=sub_train_loader.batch_size, sampler=coreset_sampler)
    coresets.append(coreset_loader)
