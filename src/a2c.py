import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from envs import ObservationBatch, VecEnv
from experiment import model_device
from rl_math import clip_grad_norm, compute_discounted_rollout_returns


@dataclass(frozen=True)
class A2CEpisode:
    env_id: int
    episode_return: float
    episode_length: int


@dataclass(frozen=True)
class A2CLog:
    step: int
    update: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    grad_norm: float | None
    rollout_steps: int
    episodes: tuple[A2CEpisode, ...] = ()


type A2CLogFn = Callable[["A2C", A2CLog], None]


@dataclass(frozen=True)
class _Rollout:
    observations: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    entropies: torch.Tensor
    next_observation: ObservationBatch
    length: int
    episodes: tuple[A2CEpisode, ...]


class A2C:
    def __init__(
        self,
        env: VecEnv,
        model: nn.Module,
        *,
        learning_rate: float,
        value_loss_coef: float,
        discount_factor: float,
        rollout_steps: int,
        max_grad_norm: float | None,
    ) -> None:
        validate_a2c_hyperparameters(
            learning_rate=learning_rate,
            value_loss_coef=value_loss_coef,
            discount_factor=discount_factor,
            rollout_steps=rollout_steps,
            max_grad_norm=max_grad_norm,
        )
        self.env = env
        self.model = model
        self.device = model_device(model)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.num_envs = env.num_envs
        self.value_loss_coef = value_loss_coef
        self.discount_factor = discount_factor
        self.rollout_steps = rollout_steps
        self.max_grad_norm = max_grad_norm
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    def train(
        self,
        num_steps: int,
        log_fn: A2CLogFn | None = None,
    ) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        observation = self.env.reset()
        assert observation.shape == (self.num_envs, *self.env.observation_shape)
        step = 0
        update = 0

        while step < num_steps:
            remaining_steps = num_steps - step
            rollout_steps = min(
                self.rollout_steps,
                max(1, math.ceil(remaining_steps / self.num_envs)),
            )
            rollout = self._collect_rollout(
                observation=observation,
                rollout_steps=rollout_steps,
            )
            observation = rollout.next_observation
            loss, policy_loss, value_loss, entropy, grad_norm = self._update(rollout)
            step += rollout.length * self.num_envs
            update += 1

            if log_fn is not None:
                log_fn(
                    self,
                    A2CLog(
                        step=step,
                        update=update,
                        loss=loss,
                        policy_loss=policy_loss,
                        value_loss=value_loss,
                        entropy=entropy,
                        grad_norm=grad_norm,
                        rollout_steps=rollout.length,
                        episodes=rollout.episodes,
                    ),
                )

    def _collect_rollout(
        self,
        *,
        observation: ObservationBatch,
        rollout_steps: int,
    ) -> _Rollout:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive.")

        observation_shape = self.env.observation_shape
        observations = torch.empty(
            (rollout_steps, self.num_envs, *observation_shape),
            dtype=torch.as_tensor(observation).dtype,
            device=self.device,
        )
        rewards = torch.empty((rollout_steps, self.num_envs), device=self.device)
        terminated = torch.empty(
            (rollout_steps, self.num_envs), dtype=torch.bool, device=self.device
        )
        truncated = torch.empty(
            (rollout_steps, self.num_envs), dtype=torch.bool, device=self.device
        )
        log_probs = torch.empty((rollout_steps, self.num_envs), device=self.device)
        values = torch.empty((rollout_steps, self.num_envs), device=self.device)
        next_values = torch.empty((rollout_steps, self.num_envs), device=self.device)
        entropies = torch.empty((rollout_steps, self.num_envs), device=self.device)
        episodes: list[A2CEpisode] = []
        previous_non_done_slots: torch.Tensor | None = None

        for rollout_index in range(rollout_steps):
            observation_tensor = torch.as_tensor(observation, device=self.device)
            assert observation_tensor.shape == (
                self.num_envs,
                *self.env.observation_shape,
            )
            observations[rollout_index] = observation_tensor
            dist, value = self.model(observation_tensor)
            assert value.shape == (self.num_envs,)
            if previous_non_done_slots is not None:
                assert rollout_index > 0
                next_values[rollout_index - 1, previous_non_done_slots] = value[
                    previous_non_done_slots
                ]

            action = dist.sample()
            step = self.env.step(action.cpu().numpy())
            assert step.observation.shape == (self.num_envs, *observation_shape)
            next_observation = np.array(step.observation, copy=True)

            rewards[rollout_index] = torch.as_tensor(
                step.reward, dtype=torch.float32, device=self.device
            )
            terminated[rollout_index] = torch.as_tensor(
                step.terminated, dtype=torch.bool, device=self.device
            )
            truncated[rollout_index] = torch.as_tensor(
                step.truncated, dtype=torch.bool, device=self.device
            )
            log_probs[rollout_index] = dist.log_prob(action)
            values[rollout_index] = value
            entropies[rollout_index] = dist.entropy()

            done = np.logical_or(step.terminated, step.truncated)
            terminated_slots = np.flatnonzero(step.terminated)
            if len(terminated_slots) > 0:
                terminated_slot_indices = torch.as_tensor(
                    terminated_slots,
                    dtype=torch.int64,
                    device=self.device,
                )
                next_values[rollout_index, terminated_slot_indices] = 0.0

            bootstrap_slots = np.flatnonzero(
                np.logical_and(step.truncated, np.logical_not(step.terminated))
            )
            if len(bootstrap_slots) > 0:
                bootstrap_slot_indices = torch.as_tensor(
                    bootstrap_slots,
                    dtype=torch.int64,
                    device=self.device,
                )
                _, bootstrap_value = self.model(
                    torch.as_tensor(
                        step.observation[bootstrap_slots],
                        device=self.device,
                    )
                )
                assert bootstrap_value.shape == (len(bootstrap_slots),)
                next_values[rollout_index, bootstrap_slot_indices] = bootstrap_value

            non_done_slots = np.flatnonzero(np.logical_not(done))
            if len(non_done_slots) > 0:
                previous_non_done_slots = torch.as_tensor(
                    non_done_slots,
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                previous_non_done_slots = None

            self._episode_returns += step.reward.astype(np.float64)
            self._episode_lengths += 1
            if np.any(done):
                done_slots = np.flatnonzero(done)
                for env_slot in done_slots:
                    episodes.append(
                        A2CEpisode(
                            env_id=int(step.env_id[env_slot]),
                            episode_return=float(self._episode_returns[env_slot]),
                            episode_length=int(self._episode_lengths[env_slot]),
                        )
                    )
                    self._episode_returns[env_slot] = 0.0
                    self._episode_lengths[env_slot] = 0

                reset_observation = self.env.reset_subset(step.env_id[done_slots])
                assert reset_observation.shape == (len(done_slots), *observation_shape)
                next_observation[done_slots] = reset_observation

            observation = next_observation

        if previous_non_done_slots is not None:
            observation_tensor = torch.as_tensor(observation, device=self.device)
            assert observation_tensor.shape == (
                self.num_envs,
                *self.env.observation_shape,
            )
            _, value = self.model(observation_tensor)
            assert value.shape == (self.num_envs,)
            next_values[rollout_steps - 1, previous_non_done_slots] = value[
                previous_non_done_slots
            ]

        return _Rollout(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            log_probs=log_probs,
            values=values,
            next_values=next_values,
            entropies=entropies,
            next_observation=observation,
            length=rollout_steps,
            episodes=tuple(episodes),
        )

    def _update(
        self, rollout: _Rollout
    ) -> tuple[float, float, float, float, float | None]:
        returns = compute_discounted_rollout_returns(
            rewards=rollout.rewards,
            next_values=rollout.next_values.detach(),
            terminated=rollout.terminated,
            truncated=rollout.truncated,
            discount_factor=self.discount_factor,
        )
        advantages = returns - rollout.values
        policy_loss = -(rollout.log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(rollout.values, returns)
        entropy = rollout.entropies.mean()
        loss = policy_loss + self.value_loss_coef * value_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return (
            float(loss.item()),
            float(policy_loss.item()),
            float(value_loss.item()),
            float(entropy.item()),
            grad_norm,
        )


def validate_a2c_hyperparameters(
    *,
    learning_rate: float,
    value_loss_coef: float,
    discount_factor: float,
    rollout_steps: int,
    max_grad_norm: float | None,
) -> None:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if value_loss_coef < 0.0:
        raise ValueError("value_loss_coef must be non-negative.")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1].")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive or None.")
