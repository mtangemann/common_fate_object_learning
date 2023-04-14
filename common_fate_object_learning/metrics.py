"""Metrics used to evaluate object models."""

import torch


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, epsilon=10e-5) -> torch.Tensor:
    mask = mask.type(torch.float32).unsqueeze(-3)

    squared_error = mask * (prediction - target) ** 2
    mask_size = 3 * mask.sum(dim=[-3, -2, -1])

    mean_squared_error = squared_error.sum(dim=[-3, -2, -1]) / mask_size
    mean_squared_error[mask_size < epsilon] = 0.0

    return mean_squared_error


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, epsilon=10e-5) -> torch.Tensor:
    mask = mask.type(torch.float32).unsqueeze(-3)

    absolute_error = mask * torch.abs(prediction - target)
    mask_size = 3 * mask.sum(dim=[-3, -2, -1])

    mean_absolute_error = absolute_error.sum(dim=[-3, -2, -1]) / mask_size
    mean_absolute_error[mask_size < epsilon] = 0.0

    return mean_absolute_error
