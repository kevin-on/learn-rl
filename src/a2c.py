import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from task_adapter import DiscreteActionTaskAdapter, State


@dataclass(frozen=True)
class A2CLog:
    step_index: int
    loss: float | None
    policy_loss: float | None
    value_loss: float | None
    grad_norm: float | None
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


type A2CLogFn[Observation, EnvAction] = Callable[
    ["A2C[Observation, EnvAction]", A2CLog],
    None,
]
type A2CEvalFn[Observation, EnvAction] = Callable[
    ["A2C[Observation, EnvAction]", int],
    None,
]


class A2C[Observation, EnvAction]:
    def __init__(
        self,
        task_adapter: DiscreteActionTaskAdapter[Observation, EnvAction],
        policy_net: nn.Module,
        value_net: nn.Module,
        *,
        policy_learning_rate: float,
        value_learning_rate: float,
        discount_factor: float,
        rollout_steps: int,
        max_grad_norm: float | None,
    ) -> None:
        validate_a2c_hyperparameters(
            policy_learning_rate=policy_learning_rate,
            value_learning_rate=value_learning_rate,
            discount_factor=discount_factor,
            rollout_steps=rollout_steps,
            max_grad_norm=max_grad_norm,
        )
        self.task_adapter = task_adapter
        self.env = task_adapter.env
        self.policy_net = policy_net
        self.value_net = value_net
        self.device = next(self.policy_net.parameters()).device
        value_device = next(self.value_net.parameters()).device
        if value_device != self.device:
            msg = (
                "policy_net and value_net must be on the same device: "
                f"got {self.device} and {value_device}"
            )
            raise ValueError(msg)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=policy_learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=value_learning_rate
        )
        self.discount_factor = discount_factor
        self.rollout_steps = rollout_steps
        self.max_grad_norm = max_grad_norm

    def train(
        self,
        num_steps: int,
        env_seed: int | None = None,
        log_fn: A2CLogFn[Observation, EnvAction] | None = None,
        eval_fn: A2CEvalFn[Observation, EnvAction] | None = None,
    ) -> None:
        observation, _info = self.env.reset(seed=env_seed)
        state = self.task_adapter.encode_observation(observation)
        states: list[State] = []
        rewards: list[float] = []
        log_probs: list[torch.Tensor] = []

        for step_index in range(num_steps):
            loss_value = None
            policy_loss_value = None
            value_loss_value = None
            grad_norm = None

            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            logits = self.policy_net(state_tensor)
            self._validate_logits(logits)
            dist = torch.distributions.Categorical(logits=logits)
            action_index = dist.sample()
            log_prob = dist.log_prob(action_index)

            env_action = self.task_adapter.action_index_to_env_action(
                int(action_index.item())
            )
            next_observation, reward, terminated, truncated, info = self.env.step(
                env_action
            )
            next_state = self.task_adapter.encode_observation(next_observation)

            states.append(state)
            rewards.append(float(reward))
            log_probs.append(log_prob.squeeze(0))
            state = next_state

            should_reset = terminated or truncated
            should_update = (
                len(states) == self.rollout_steps
                or should_reset
                or step_index == num_steps - 1
            )
            if should_update:
                policy_loss, value_loss = self._compute_losses(
                    states=states,
                    rewards=rewards,
                    log_probs=log_probs,
                    rollout_end_state=state,
                    terminal=terminated,
                )
                loss = policy_loss + value_loss
                loss_value = float(loss.item())
                policy_loss_value = float(policy_loss.item())
                value_loss_value = float(value_loss.item())

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    clipped_norm = torch.nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.policy_net.parameters(),
                            self.value_net.parameters(),
                        ),
                        max_norm=self.max_grad_norm,
                    )
                    grad_norm = float(
                        clipped_norm.item()
                        if isinstance(clipped_norm, torch.Tensor)
                        else clipped_norm
                    )
                self.policy_optimizer.step()
                self.value_optimizer.step()
                states.clear()
                rewards.clear()
                log_probs.clear()

            if should_reset:
                observation, _info = self.env.reset()
                state = self.task_adapter.encode_observation(observation)

            if log_fn is not None:
                log_fn(
                    self,
                    A2CLog(
                        step_index=step_index,
                        loss=loss_value,
                        policy_loss=policy_loss_value,
                        value_loss=value_loss_value,
                        grad_norm=grad_norm,
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    ),
                )
            if eval_fn is not None:
                eval_fn(self, step_index)

    def _compute_losses(
        self,
        *,
        states: list[State],
        rewards: list[float],
        log_probs: list[torch.Tensor],
        rollout_end_state: State,
        terminal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states_tensor = torch.as_tensor(
            np.asarray(states), dtype=torch.float32, device=self.device
        )
        state_values = self.value_net(states_tensor).squeeze(-1)
        if state_values.shape != (len(states),):
            msg = (
                "Value network must return one value per state: "
                f"expected shape ({len(states)},) or ({len(states)}, 1), "
                f"got {state_values.shape}"
            )
            raise ValueError(msg)

        returns = self._compute_rollout_returns(
            rewards=rewards,
            rollout_end_state=rollout_end_state,
            terminal=terminal,
        )
        advantages = returns - state_values
        log_probs_tensor = torch.stack(log_probs).reshape_as(advantages)

        policy_loss = -(log_probs_tensor * advantages.detach()).mean()
        value_loss = F.mse_loss(state_values, returns)
        return policy_loss, value_loss

    @torch.no_grad()
    def _compute_rollout_returns(
        self,
        *,
        rewards: list[float],
        rollout_end_state: State,
        terminal: bool,
    ) -> torch.Tensor:
        if terminal:
            rollout_return = torch.zeros((), dtype=torch.float32, device=self.device)
        else:
            state_tensor = torch.as_tensor(
                rollout_end_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            rollout_return = self.value_net(state_tensor).reshape(())

        returns: list[torch.Tensor] = []
        for reward in reversed(rewards):
            rollout_return = (
                torch.as_tensor(reward, dtype=torch.float32, device=self.device)
                + self.discount_factor * rollout_return
            )
            returns.append(rollout_return)

        returns.reverse()
        return torch.stack(returns)

    def _validate_logits(self, logits: torch.Tensor) -> None:
        if logits.ndim != 2:
            msg = f"Policy network must return a 2D tensor, got shape {logits.shape}"
            raise ValueError(msg)

        if logits.shape[1] != self.task_adapter.num_actions:
            msg = (
                "Policy network action dimension must match the task adapter: "
                f"expected {self.task_adapter.num_actions}, got {logits.shape[1]}"
            )
            raise ValueError(msg)


def validate_a2c_hyperparameters(
    *,
    policy_learning_rate: float,
    value_learning_rate: float,
    discount_factor: float,
    rollout_steps: int,
    max_grad_norm: float | None,
) -> None:
    if policy_learning_rate <= 0.0:
        raise ValueError("policy_learning_rate must be positive.")
    if value_learning_rate <= 0.0:
        raise ValueError("value_learning_rate must be positive.")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1].")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive or None.")
