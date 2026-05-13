import math

import torch


def compute_discounted_returns(
    *,
    rewards: torch.Tensor,
    bootstrap_value: torch.Tensor,
    discount_factor: float,
) -> torch.Tensor:
    if rewards.ndim < 1:
        raise ValueError("rewards must have at least one dimension.")
    if bootstrap_value.shape != rewards.shape[1:]:
        msg = (
            "bootstrap_value shape must match rewards without rollout dimension: "
            f"expected {rewards.shape[1:]}, got {bootstrap_value.shape}"
        )
        raise ValueError(msg)
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1].")

    returns = torch.empty_like(rewards)
    rollout_return = bootstrap_value
    for step in reversed(range(rewards.shape[0])):
        rollout_return = rewards[step] + discount_factor * rollout_return
        returns[step] = rollout_return
    return returns


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    discount_factor: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.shape != values.shape:
        raise ValueError("values must have the same shape as rewards.")
    if rewards.shape != next_values.shape:
        raise ValueError("next_values must have the same shape as rewards.")
    if terminated.shape != rewards.shape:
        raise ValueError("terminated must have the same shape as rewards.")
    if truncated.shape != rewards.shape:
        raise ValueError("truncated must have the same shape as rewards.")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1].")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1].")

    advantages = torch.empty_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1:], device=rewards.device)
    for step in reversed(range(rewards.shape[0])):
        episode_done = torch.logical_or(terminated[step], truncated[step])
        recurrence_mask = (~episode_done).to(dtype=torch.float32)
        bootstrap_mask = (~terminated[step]).to(dtype=torch.float32)
        delta = (
            rewards[step]
            + discount_factor * next_values[step] * bootstrap_mask
            - values[step]
        )
        last_advantage = (
            delta + discount_factor * gae_lambda * recurrence_mask * last_advantage
        )
        advantages[step] = last_advantage

    returns = advantages + values
    return advantages, returns


def compute_discounted_rollout_returns(
    *,
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    discount_factor: float,
) -> torch.Tensor:
    if rewards.shape != next_values.shape:
        raise ValueError("next_values must have the same shape as rewards.")
    if terminated.shape != rewards.shape:
        raise ValueError("terminated must have the same shape as rewards.")
    if truncated.shape != rewards.shape:
        raise ValueError("truncated must have the same shape as rewards.")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1].")

    returns = torch.empty_like(rewards)
    rollout_return = next_values[-1]
    for step in reversed(range(rewards.shape[0])):
        rollout_return = torch.where(
            terminated[step],
            torch.zeros_like(rollout_return),
            torch.where(truncated[step], next_values[step], rollout_return),
        )
        rollout_return = rewards[step] + discount_factor * rollout_return
        returns[step] = rollout_return
    return returns


@torch.no_grad()
def clip_grad_norm(
    parameters: object,
    max_grad_norm: float | None,
) -> float | None:
    if max_grad_norm is None:
        return None
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive or None.")

    clipped_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)
    return float(
        clipped_norm.item() if isinstance(clipped_norm, torch.Tensor) else clipped_norm
    )


@torch.no_grad()
def clip_grad_tensors(
    grads: list[torch.Tensor | None],
    max_grad_norm: float | None,
) -> float | None:
    if max_grad_norm is None:
        return None
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive or None.")

    grad_tensors = [grad for grad in grads if grad is not None]
    if not grad_tensors:
        return 0.0

    total_norm = math.sqrt(
        sum(
            float(torch.sum(grad.detach() * grad.detach()).item())
            for grad in grad_tensors
        )
    )
    if total_norm > max_grad_norm:
        scale = max_grad_norm / (total_norm + 1e-6)
        for grad in grad_tensors:
            grad.mul_(scale)

    return total_norm
