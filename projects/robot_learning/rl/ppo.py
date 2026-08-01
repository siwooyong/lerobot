"""Chunk-aware PPO mathematics."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PPOLoss:
    policy: Tensor
    value: Tensor
    entropy: Tensor
    kl: Tensor
    clipped: Tensor


def gae(rewards: Tensor, dones: Tensor, values: Tensor, worker_ids: Tensor, bootstrap: dict[int, Tensor], gamma: float, lam: float) -> tuple[Tensor, Tensor]:
    """Compute GAE per worker, treating each fixed action chunk as one PPO step."""
    advantage = torch.zeros_like(rewards)
    for worker in worker_ids.unique().tolist():
        indices = (worker_ids == worker).nonzero(as_tuple=True)[0]
        carry, next_value = torch.zeros((), device=values.device), bootstrap[int(worker)]
        for index in reversed(indices.tolist()):
            alive = (~dones[index]).to(values.dtype)
            delta = rewards[index] + gamma * next_value * alive - values[index]
            carry = delta + gamma * lam * alive * carry
            advantage[index], next_value = carry, values[index]
    return advantage, advantage + values


def normalize_advantage(advantage: Tensor, task_ids: Tensor) -> Tensor:
    """Normalize advantages independently for each task in a PPO rollout batch."""
    normalized = torch.empty_like(advantage)
    for task_id in task_ids.unique().tolist():
        indices = task_ids == task_id
        task_advantage = advantage[indices]
        normalized[indices] = (task_advantage - task_advantage.mean()) / task_advantage.std(unbiased=False).clamp_min(1e-4)
    return normalized


def _huber(error: Tensor, delta: float) -> Tensor:
    return torch.where(error.abs() < delta, 0.5 * error.square(), delta * (error.abs() - 0.5 * delta))


def loss(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantage: Tensor,
    value: Tensor,
    old_value: Tensor,
    target: Tensor,
    clip: float,
    value_clip: float,
    huber_delta: float,
    entropy: Tensor,
    entropy_bonus: float,
) -> PPOLoss:
    ratio = (new_log_prob - old_log_prob).exp()
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    clipped = torch.where(advantage >= 0, ratio > 1 + clip, ratio < 1 - clip)
    value_clipped = old_value + (value - old_value).clamp(-value_clip, value_clip)
    value_loss = torch.maximum(_huber(target - value, huber_delta), _huber(target - value_clipped, huber_delta))
    entropy_loss = entropy.mean()
    return PPOLoss(
        -objective.mean() - entropy_bonus * entropy_loss,
        value_loss.mean(),
        entropy_loss,
        (old_log_prob - new_log_prob).mean(),
        clipped,
    )
