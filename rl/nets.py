"""Fully connected networks (replaces MimicKit's learning/nets/fc_2layers_1024units.py)."""

import torch

HIDDEN_SIZES = (1024, 512)        # article, section 6.2: two hidden layers, 1024 and 512 ReLU units


def mlp(in_size, hidden_sizes=HIDDEN_SIZES, activation=torch.nn.ReLU):
    """Linear -> activation for each hidden size, biases set to zero, weights with PyTorch's
    default init (as MimicKit). Returns a torch.nn.Sequential; its output size is hidden_sizes[-1]."""
    layers = []
    for out_size in hidden_sizes:
        layer = torch.nn.Linear(in_size, out_size)
        torch.nn.init.zeros_(layer.bias)
        layers += [layer, activation()]
        in_size = out_size
    return torch.nn.Sequential(*layers)
