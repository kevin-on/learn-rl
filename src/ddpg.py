import copy
import math
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from envs import BoxActionSpec, ObservationBatch, VecEnv
from experiment import model_device

type State = np.ndarray
type Action = np.ndarray
type Experience = tuple[State, Action, float, State, bool]


@dataclass(frozen=True)
class DDPGEpisode:
    env_id: int
    episode_return: float
    episode_length: int


@dataclass(frozen=True)
class DDPGUpdateStats:
    actor_loss: float
    critic_loss: float
    q_mean: float
    target_q_mean: float


@dataclass(frozen=True)
class DDPGLog:
    step: int
    update: int
    stats: DDPGUpdateStats | None
    reward_mean: float
    episodes: tuple[DDPGEpisode, ...] = ()


type DDPGLogFn = Callable[["DDPG", DDPGLog], None]


class OrnsteinUhlenbeckNoise:
    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1.0,
    ) -> None:
        if any(size <= 0 for size in shape):
            raise ValueError("shape dimensions must be positive.")
        if theta <= 0.0:
            raise ValueError("theta must be positive.")
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        self.shape = shape
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.state = np.zeros(shape, dtype=np.float32)

    def reset(self, indices: np.ndarray | None = None) -> None:
        if indices is None:
            self.state.fill(0.0)
            return

        self.state[np.asarray(indices, dtype=np.int64)] = 0.0

    def sample(self) -> np.ndarray:
        noise = np.random.normal(size=self.shape).astype(np.float32)
        dx = (
            self.theta * -self.state * self.dt + self.sigma * math.sqrt(self.dt) * noise
        )
        self.state = self.state + dx
        return np.array(self.state, copy=True)


class DDPG:
    def __init__(
        self,
        env: VecEnv,
        model: nn.Module,
        *,
        actor_learning_rate: float,
        critic_learning_rate: float,
        critic_weight_decay: float,
        discount_factor: float,
        soft_update_rate: float,
        buffer_capacity: int,
        batch_size: int,
        ou_theta: float = 0.15,
        ou_sigma: float = 0.2,
    ) -> None:
        validate_ddpg_hyperparameters(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            critic_weight_decay=critic_weight_decay,
            discount_factor=discount_factor,
            soft_update_rate=soft_update_rate,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
        )
        if not isinstance(env.action_spec, BoxActionSpec):
            raise ValueError("DDPG requires a Box action space.")
        if not hasattr(model, "actor") or not hasattr(model, "critic"):
            raise TypeError("DDPG model must expose actor and critic submodules.")
        if not callable(getattr(model, "act", None)) or not callable(
            getattr(model, "q", None)
        ):
            raise TypeError("DDPG model must expose act() and q() methods.")

        self.env = env
        self.online_model = model
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        self.device = model_device(self.online_model)
        self.actor_optimizer = torch.optim.Adam(
            self.online_model.actor.parameters(),
            lr=actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.online_model.critic.parameters(),
            lr=critic_learning_rate,
            weight_decay=critic_weight_decay,
        )
        self.num_envs = env.num_envs
        self.action_spec = env.action_spec
        self.discount_factor = discount_factor
        self.soft_update_rate = soft_update_rate
        self.batch_size = batch_size
        self.replay_buffer: deque[Experience] = deque(maxlen=buffer_capacity)
        self.step = 0
        self.update = 0
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._action_low = np.asarray(self.action_spec.low, dtype=np.float32)
        self._action_high = np.asarray(self.action_spec.high, dtype=np.float32)
        self._action_scale = (self._action_high - self._action_low) / 2.0
        self._ou_noise = OrnsteinUhlenbeckNoise(
            shape=(self.num_envs, *self.action_spec.shape),
            theta=ou_theta,
            sigma=ou_sigma,
        )

    def train(
        self,
        num_steps: int,
        log_fn: DDPGLogFn | None = None,
    ) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        observation = self.env.reset()
        self._ou_noise.reset()
        assert observation.shape == (self.num_envs, *self.env.observation_shape)

        while self.step < num_steps:
            actions = self._select_actions(observation)
            env_step = self.env.step(actions)
            assert env_step.observation.shape == (
                self.num_envs,
                *self.env.observation_shape,
            )
            next_observation = np.array(env_step.observation, copy=True)
            done = np.logical_or(env_step.terminated, env_step.truncated)

            for env_slot in range(self.num_envs):
                self.replay_buffer.append(
                    (
                        np.array(observation[env_slot], copy=True),
                        np.array(actions[env_slot], copy=True),
                        float(env_step.reward[env_slot]),
                        np.array(env_step.observation[env_slot], copy=True),
                        bool(env_step.terminated[env_slot]),
                    )
                )

            episodes = self._record_episodes(env_step.reward, done, env_step.env_id)
            if np.any(done):
                done_slots = np.flatnonzero(done)
                self._ou_noise.reset(done_slots)
                reset_observation = self.env.reset_subset(env_step.env_id[done_slots])
                assert reset_observation.shape == (
                    len(done_slots),
                    *self.env.observation_shape,
                )
                next_observation[done_slots] = reset_observation
            observation = next_observation

            stats = None
            if len(self.replay_buffer) >= self.batch_size:
                stats = self._update()
                self.update += 1

            self.step += self.num_envs
            if log_fn is not None:
                log_fn(
                    self,
                    DDPGLog(
                        step=self.step,
                        update=self.update,
                        stats=stats,
                        reward_mean=float(np.mean(env_step.reward)),
                        episodes=tuple(episodes),
                    ),
                )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "update": self.update,
            "model_state": self.online_model.state_dict(),
            "optimizer_state": {
                "actor": self.actor_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict(),
            },
            "algorithm_state": {
                "target_model_state": self.target_model.state_dict(),
                "replay_buffer": list(self.replay_buffer),
            },
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self.online_model.load_state_dict(state["model_state"])
        optimizer_state = state["optimizer_state"]
        self.actor_optimizer.load_state_dict(optimizer_state["actor"])
        self.critic_optimizer.load_state_dict(optimizer_state["critic"])
        algorithm_state = state["algorithm_state"]
        self.target_model.load_state_dict(algorithm_state["target_model_state"])
        self.replay_buffer = deque(
            algorithm_state["replay_buffer"],
            maxlen=self.replay_buffer.maxlen,
        )
        self.step = int(state["step"])
        self.update = int(state["update"])
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        self._ou_noise.reset()

    @torch.no_grad()
    def _select_actions(
        self,
        observation: ObservationBatch,
    ) -> np.ndarray:
        observation_tensor = torch.as_tensor(observation, device=self.device)
        actions = self.online_model.act(observation_tensor).cpu().numpy()
        actions = actions + self._ou_noise.sample() * self._action_scale
        return np.clip(actions, self._action_low, self._action_high).astype(
            self.action_spec.dtype,
            copy=False,
        )

    def _update(self) -> DDPGUpdateStats:
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch, strict=False)
        states = torch.as_tensor(np.asarray(states), device=self.device)
        actions = torch.as_tensor(
            np.asarray(actions),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_states = torch.as_tensor(np.asarray(next_states), device=self.device)
        terminals = torch.as_tensor(
            terminals,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.target_model.act(next_states)
            next_q_values = self.target_model.q(next_states, next_actions)
            target_q_values = (
                rewards + self.discount_factor * (1.0 - terminals) * next_q_values
            )

        q_values = self.online_model.q(states, actions)
        critic_loss = F.mse_loss(q_values, target_q_values)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        set_requires_grad(self.online_model.critic, False)
        try:
            policy_actions = self.online_model.act(states)
            actor_loss = -self.online_model.q(states, policy_actions).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            set_requires_grad(self.online_model.critic, True)

        soft_update(
            self.target_model,
            self.online_model,
            self.soft_update_rate,
        )

        return DDPGUpdateStats(
            actor_loss=float(actor_loss.item()),
            critic_loss=float(critic_loss.item()),
            q_mean=float(q_values.detach().mean().item()),
            target_q_mean=float(target_q_values.detach().mean().item()),
        )

    def _record_episodes(
        self,
        rewards: np.ndarray,
        done: np.ndarray,
        env_ids: np.ndarray,
    ) -> list[DDPGEpisode]:
        self._episode_returns += rewards.astype(np.float64)
        self._episode_lengths += 1

        episodes: list[DDPGEpisode] = []
        for env_slot in np.flatnonzero(done):
            episodes.append(
                DDPGEpisode(
                    env_id=int(env_ids[env_slot]),
                    episode_return=float(self._episode_returns[env_slot]),
                    episode_length=int(self._episode_lengths[env_slot]),
                )
            )
            self._episode_returns[env_slot] = 0.0
            self._episode_lengths[env_slot] = 0
        return episodes


def validate_ddpg_hyperparameters(
    *,
    actor_learning_rate: float,
    critic_learning_rate: float,
    critic_weight_decay: float,
    discount_factor: float,
    soft_update_rate: float,
    buffer_capacity: int,
    batch_size: int,
) -> None:
    if actor_learning_rate <= 0.0:
        raise ValueError("actor_learning_rate must be positive.")
    if critic_learning_rate <= 0.0:
        raise ValueError("critic_learning_rate must be positive.")
    if critic_weight_decay < 0.0:
        raise ValueError("critic_weight_decay must be non-negative.")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1].")
    if not 0.0 <= soft_update_rate <= 1.0:
        raise ValueError("soft_update_rate must be in [0, 1].")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if buffer_capacity < batch_size:
        raise ValueError("buffer_capacity must be at least batch_size.")


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


@torch.no_grad()
def soft_update(
    target_model: nn.Module,
    online_model: nn.Module,
    soft_update_rate: float,
) -> None:
    for target_param, online_param in zip(
        target_model.parameters(),
        online_model.parameters(),
        strict=True,
    ):
        target_param.mul_(1.0 - soft_update_rate).add_(
            online_param,
            alpha=soft_update_rate,
        )
