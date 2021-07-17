## Variational Continual Learning in PyTorch

Re-Implementation of [Variational Continual Learning](https://arxiv.org/abs/1710.10628) by  Nguyen, Li, Bui, and Turner (ICLR 2018).

### SplitMNIST: VCL v/s Baselines
![SplitMNIST Baselines](plots/splitmnist_baselines.png)

### PermutedMNIST: VCL v/s Baselines
![PermutedMNIST Baselines](plots/permutedmnist_baselines.png)

### Vary beta (Weight coefficient for KL term)
Varying beta while keeping memory fixed at 200 samples per task.

![Vary beta SplitMNIST](plots/splitmnist_fixed_memory_vary_beta.png)
![Vary beta PermutedMNIST](plots/permutedmnist_fixed_memory_vary_beta.png)

### Vary Memory
Varying memory while keeping beta fixed at 1.

![Vary Memory SplitMNIST](plots/splitmnist_fixed_beta_vary_memory.png)
![Vary Memory PermutedMNIST](plots/permutedmnist_fixed_beta_vary_memory.png)

### Vary Memory (Fixed prior)
Varying memory while keeping prior fixed to Gaussian distribution.

![Vary Memory SplitMNIST](plots/splitmnist_constant_KL_vary_memory.png)
![Vary Memory PermutedMNIST](plots/permutedmnist_constant_KL_vary_memory.png)
