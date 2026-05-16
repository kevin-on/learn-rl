import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from envs import BoxActionSpec, DiscreteActionSpec, ObservationBatch, VecEnv
from experiment import model_device
from rl_math import clip_grad_norm, compute_gae


@dataclass(frozen=True)
class PPOEpisode:
    env_id: int
    episode_return: float
    episode_length: int


@dataclass(frozen=True)
class PPOUpdateStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float
    grad_norm: float | None


@dataclass(frozen=True)
class PPOLog:
    step: int
    update: int
    stats: PPOUpdateStats
    rollout_steps: int
    episodes: tuple[PPOEpisode, ...] = ()


type PPOLogFn = Callable[["PPO", PPOLog], None]


@dataclass(frozen=True)
class _Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    next_observation: ObservationBatch
    length: int
    episodes: tuple[PPOEpisode, ...]


class PPO:
    def __init__(
        self,
        env: VecEnv,
        model: nn.Module,
        *,
        learning_rate: float,
        rollout_steps: int,
        minibatch_size: int,
        epochs: int,
        discount_factor: float,
        gae_lambda: float,
        clip_coef: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float | None,
    ) -> None:
        validate_ppo_hyperparameters(
            learning_rate=learning_rate,
            num_envs=env.num_envs,
            rollout_steps=rollout_steps,
            minibatch_size=minibatch_size,
            epochs=epochs,
            discount_factor=discount_factor,
            gae_lambda=gae_lambda,
            clip_coef=clip_coef,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )
        self.env = env
        self.model = model
        self.device = model_device(model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_envs = env.num_envs
        self.action_spec = env.action_spec
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.step = 0
        self.update = 0
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    def train(self, num_steps: int, log_fn: PPOLogFn | None = None) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        observation = self.env.reset()
        assert observation.shape == (self.num_envs, *self.env.observation_shape)

        while self.step < num_steps:
            remaining_steps = num_steps - self.step
            rollout_steps = min(
                self.rollout_steps,
                max(1, math.ceil(remaining_steps / self.num_envs)),
            )
            rollout = self._collect_rollout(
                observation=observation,
                rollout_steps=rollout_steps,
            )
            observation = rollout.next_observation
            stats = self._update(rollout)
            self.step += rollout.length * self.num_envs
            self.update += 1

            if log_fn is not None:
                log_fn(
                    self,
                    PPOLog(
                        step=self.step,
                        update=self.update,
                        stats=stats,
                        rollout_steps=rollout.length,
                        episodes=rollout.episodes,
                    ),
                )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "update": self.update,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "algorithm_state": {},
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.step = int(state["step"])
        self.update = int(state["update"])
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)

    @torch.no_grad()
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
        actions = _empty_action_rollout(
            action_spec=self.action_spec,
            rollout_steps=rollout_steps,
            num_envs=self.num_envs,
            device=self.device,
        )
        rewards = torch.empty((rollout_steps, self.num_envs), device=self.device)
        terminated = torch.empty(
            (rollout_steps, self.num_envs), dtype=torch.bool, device=self.device
        )
        truncated = torch.empty(
            (rollout_steps, self.num_envs), dtype=torch.bool, device=self.device
        )
        old_log_probs = torch.empty((rollout_steps, self.num_envs), device=self.device)
        values = torch.empty((rollout_steps, self.num_envs), device=self.device)
        next_values = torch.empty((rollout_steps, self.num_envs), device=self.device)
        episodes: list[PPOEpisode] = []
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
            log_prob = dist.log_prob(action)
            step = self.env.step(action.cpu().numpy())

            assert step.observation.shape == (self.num_envs, *observation_shape)
            assert step.reward.shape == (self.num_envs,)
            assert step.terminated.shape == (self.num_envs,)
            assert step.truncated.shape == (self.num_envs,)
            assert step.env_id.shape == (self.num_envs,)
            next_observation = np.array(step.observation, copy=True)

            actions[rollout_index] = action
            rewards[rollout_index] = torch.as_tensor(
                step.reward, dtype=torch.float32, device=self.device
            )
            terminated[rollout_index] = torch.as_tensor(
                step.terminated, dtype=torch.bool, device=self.device
            )
            truncated[rollout_index] = torch.as_tensor(
                step.truncated, dtype=torch.bool, device=self.device
            )
            old_log_probs[rollout_index] = log_prob
            values[rollout_index] = value

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
                        PPOEpisode(
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
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            old_log_probs=old_log_probs,
            values=values,
            next_values=next_values,
            next_observation=observation,
            length=rollout_steps,
            episodes=tuple(episodes),
        )

    def _update(self, rollout: _Rollout) -> PPOUpdateStats:
        assert rollout.rewards.shape == (rollout.length, self.num_envs)
        assert rollout.observations.shape[:2] == rollout.rewards.shape
        assert rollout.actions.shape[:2] == rollout.rewards.shape
        assert rollout.terminated.shape == rollout.rewards.shape
        assert rollout.truncated.shape == rollout.rewards.shape
        assert rollout.old_log_probs.shape == rollout.rewards.shape
        assert rollout.values.shape == rollout.rewards.shape
        assert rollout.next_values.shape == rollout.rewards.shape

        advantages, returns = compute_gae(
            rewards=rollout.rewards,
            values=rollout.values,
            next_values=rollout.next_values,
            terminated=rollout.terminated,
            truncated=rollout.truncated,
            discount_factor=self.discount_factor,
            gae_lambda=self.gae_lambda,
        )
        return self._update_model(
            rollout=rollout,
            advantages=advantages,
            returns=returns,
        )

    def _update_model(
        self,
        *,
        rollout: _Rollout,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> PPOUpdateStats:
        num_rollout_steps, num_envs = rollout.rewards.shape
        num_transitions = num_rollout_steps * num_envs
        flat_observations = rearrange(
            rollout.observations,
            "rollout_step env ... -> (rollout_step env) ...",
        )
        actions = rearrange(
            rollout.actions,
            "rollout_step env ... -> (rollout_step env) ...",
        )
        old_log_probs = rearrange(
            rollout.old_log_probs, "rollout_step env -> (rollout_step env)"
        )
        flat_advantages = rearrange(
            advantages, "rollout_step env -> (rollout_step env)"
        )
        flat_returns = rearrange(returns, "rollout_step env -> (rollout_step env)")

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_seen = 0
        last_grad_norm: float | None = None
        minibatch_size = min(self.minibatch_size, num_transitions)

        for _epoch in range(self.epochs):
            for minibatch_indices in make_minibatch_indices(
                num_transitions,
                minibatch_size,
                device=self.device,
            ):
                dist, values = self.model(flat_observations[minibatch_indices])
                assert values.shape == (len(minibatch_indices),)

                new_log_probs = dist.log_prob(actions[minibatch_indices])
                entropy = dist.entropy().mean()
                ratio = (new_log_probs - old_log_probs[minibatch_indices]).exp()
                minibatch_advantages = flat_advantages[minibatch_indices]
                unclipped_loss = ratio * minibatch_advantages
                clipped_loss = (
                    torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    * minibatch_advantages
                )
                policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()
                value_loss = F.mse_loss(values, flat_returns[minibatch_indices])
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                last_grad_norm = clip_grad_norm(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.clip_coef)
                        .to(dtype=torch.float32)
                        .mean()
                    )
                    seen = len(minibatch_indices)
                    total_loss += float(loss.item()) * seen
                    total_policy_loss += float(policy_loss.item()) * seen
                    total_value_loss += float(value_loss.item()) * seen
                    total_entropy += float(entropy.item()) * seen
                    total_clip_fraction += float(clip_fraction.item()) * seen
                    total_seen += seen

        if total_seen == 0:
            raise RuntimeError("PPO update did not process any minibatches.")

        return PPOUpdateStats(
            loss=total_loss / total_seen,
            policy_loss=total_policy_loss / total_seen,
            value_loss=total_value_loss / total_seen,
            entropy=total_entropy / total_seen,
            clip_fraction=total_clip_fraction / total_seen,
            grad_norm=last_grad_norm,
        )


def make_minibatch_indices(
    num_transitions: int,
    minibatch_size: int,
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    if num_transitions <= 0:
        raise ValueError("num_transitions must be positive.")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive.")
    if minibatch_size > num_transitions:
        raise ValueError("minibatch_size must be at most num_transitions.")

    indices = torch.randperm(num_transitions, device=device)
    return [
        indices[start : start + minibatch_size]
        for start in range(0, num_transitions, minibatch_size)
    ]


def _empty_action_rollout(
    *,
    action_spec: DiscreteActionSpec | BoxActionSpec,
    rollout_steps: int,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(action_spec, DiscreteActionSpec):
        return torch.empty(
            (rollout_steps, num_envs),
            dtype=torch.int64,
            device=device,
        )
    if isinstance(action_spec, BoxActionSpec):
        return torch.empty(
            (rollout_steps, num_envs, *action_spec.shape),
            dtype=torch.float32,
            device=device,
        )

    msg = f"Unsupported action spec: {type(action_spec).__name__}."
    raise TypeError(msg)


def validate_ppo_hyperparameters(
    *,
    learning_rate: float,
    minibatch_size: int,
    epochs: int,
    discount_factor: float,
    gae_lambda: float,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float | None,
    num_envs: int | None = None,
    rollout_steps: int | None = None,
) -> None:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if num_envs is not None and num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if rollout_steps is not None and rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive.")
    if (
        num_envs is not None
        and rollout_steps is not None
        and minibatch_size > num_envs * rollout_steps
    ):
        raise ValueError("minibatch_size must be at most num_envs * rollout_steps.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1].")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1].")
    if clip_coef <= 0.0:
        raise ValueError("clip_coef must be positive.")
    if value_coef < 0.0:
        raise ValueError("value_coef must be non-negative.")
    if entropy_coef < 0.0:
        raise ValueError("entropy_coef must be non-negative.")
    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive or None.")
