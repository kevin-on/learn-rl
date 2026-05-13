import copy
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from envs import DiscreteVecEnv, ObservationBatch
from experiment import model_device
from rl_math import clip_grad_norm

type State = np.ndarray

# Replay-buffer transition: state, action, reward, next state, terminal flag.
type Experience = tuple[State, int, float, State, bool]


@dataclass(frozen=True)
class DQNEpisode:
    env_id: int
    episode_return: float
    episode_length: int


@dataclass(frozen=True)
class DQNLog:
    step: int
    loss: float | None
    grad_norm: float | None
    reward_mean: float
    exploration_rate: float
    episodes: tuple[DQNEpisode, ...] = ()


type DQNLogFn = Callable[["DQN", DQNLog], None]
type DQNEvalFn = Callable[["DQN", int, float], None]
type ExplorationRateFn = Callable[[int], float]


class DQN:
    def __init__(
        self,
        env: DiscreteVecEnv,
        q_net: nn.Module,
        *,
        learning_rate: float,
        discount_factor: float,
        soft_update_rate: float,
        buffer_capacity: int,
        batch_size: int,
        learning_starts: int,
        max_grad_norm: float | None,
    ) -> None:
        validate_dqn_hyperparameters(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            soft_update_rate=soft_update_rate,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            learning_starts=learning_starts,
            max_grad_norm=max_grad_norm,
        )
        if env.num_actions <= 0:
            raise ValueError("env.num_actions must be positive.")

        self.env = env
        self.online_q_net = q_net
        self.target_q_net = copy.deepcopy(q_net)
        self.target_q_net.eval()
        self.device = model_device(self.online_q_net)
        self.optimizer = torch.optim.Adam(
            self.online_q_net.parameters(), lr=learning_rate
        )
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.discount_factor = discount_factor
        self.soft_update_rate = soft_update_rate
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.max_grad_norm = max_grad_norm
        self.replay_buffer: deque[Experience] = deque(maxlen=buffer_capacity)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    def train(
        self,
        num_steps: int,
        exploration_rate_fn: ExplorationRateFn,
        log_fn: DQNLogFn | None = None,
        eval_fn: DQNEvalFn | None = None,
    ) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        observation = self.env.reset()
        assert observation.shape == (self.num_envs, *self.env.observation_shape)
        step = 0

        while step < num_steps:
            exploration_rate = validate_exploration_rate(exploration_rate_fn(step))
            action_indices = self._select_actions(observation, exploration_rate)
            env_step = self.env.step(action_indices)
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
                        int(action_indices[env_slot]),
                        float(env_step.reward[env_slot]),
                        np.array(env_step.observation[env_slot], copy=True),
                        bool(env_step.terminated[env_slot]),
                    )
                )

            episodes = self._record_episodes(env_step.reward, done, env_step.env_id)
            if np.any(done):
                done_slots = np.flatnonzero(done)
                reset_observation = self.env.reset_subset(env_step.env_id[done_slots])
                assert reset_observation.shape == (
                    len(done_slots),
                    *self.env.observation_shape,
                )
                next_observation[done_slots] = reset_observation
            observation = next_observation

            loss_value = None
            grad_norm = None
            if (
                step + self.num_envs > self.learning_starts
                and len(self.replay_buffer) >= self.batch_size
            ):
                loss = self._compute_td_loss()
                loss_value = float(loss.item())

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm(
                    self.online_q_net.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                soft_update(
                    self.target_q_net,
                    self.online_q_net,
                    self.soft_update_rate,
                )

            step += self.num_envs
            if log_fn is not None:
                log_fn(
                    self,
                    DQNLog(
                        step=step,
                        loss=loss_value,
                        grad_norm=grad_norm,
                        reward_mean=float(np.mean(env_step.reward)),
                        exploration_rate=exploration_rate,
                        episodes=tuple(episodes),
                    ),
                )
            if eval_fn is not None:
                eval_fn(self, step, exploration_rate)

    def _select_actions(
        self, observation: ObservationBatch, exploration_rate: float
    ) -> np.ndarray:
        random_actions = np.random.randint(
            self.num_actions, size=self.num_envs, dtype=np.int32
        )
        explore = np.random.random(self.num_envs) < exploration_rate
        if np.all(explore):
            return random_actions

        observation_tensor = torch.as_tensor(observation, device=self.device)
        with torch.no_grad():
            q_values = self.online_q_net(observation_tensor)
            assert q_values.shape == (self.num_envs, self.num_actions)
        greedy_actions = q_values.argmax(dim=1).cpu().numpy().astype(np.int32)
        return np.where(explore, random_actions, greedy_actions).astype(np.int32)

    def _compute_td_loss(self) -> torch.Tensor:
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, action_indices, rewards, next_states, terminals = zip(
            *batch, strict=False
        )
        states = torch.as_tensor(np.asarray(states), device=self.device)
        action_indices = torch.as_tensor(
            action_indices, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states = torch.as_tensor(np.asarray(next_states), device=self.device)
        terminals = torch.as_tensor(
            terminals, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        q_values = self.online_q_net(states)
        q_values = q_values.gather(1, action_indices)

        with torch.no_grad():
            next_online_q_values = self.online_q_net(next_states)
            next_action_indices = next_online_q_values.argmax(dim=1, keepdim=True)
            next_target_q_values = self.target_q_net(next_states)
            next_q_values = next_target_q_values.gather(1, next_action_indices)
            target_q_values = (
                rewards + self.discount_factor * (1 - terminals) * next_q_values
            )

        return F.smooth_l1_loss(q_values, target_q_values)

    def _record_episodes(
        self,
        rewards: np.ndarray,
        done: np.ndarray,
        env_ids: np.ndarray,
    ) -> list[DQNEpisode]:
        self._episode_returns += rewards.astype(np.float64)
        self._episode_lengths += 1

        episodes: list[DQNEpisode] = []
        for env_slot in np.flatnonzero(done):
            episodes.append(
                DQNEpisode(
                    env_id=int(env_ids[env_slot]),
                    episode_return=float(self._episode_returns[env_slot]),
                    episode_length=int(self._episode_lengths[env_slot]),
                )
            )
            self._episode_returns[env_slot] = 0.0
            self._episode_lengths[env_slot] = 0
        return episodes


def validate_exploration_rate(exploration_rate: float) -> float:
    exploration_rate = float(exploration_rate)
    if not 0.0 <= exploration_rate <= 1.0:
        msg = f"exploration rate must be in [0, 1], got {exploration_rate}"
        raise ValueError(msg)
    return exploration_rate


def validate_dqn_hyperparameters(
    *,
    learning_rate: float,
    discount_factor: float,
    soft_update_rate: float,
    buffer_capacity: int,
    batch_size: int,
    learning_starts: int,
    max_grad_norm: float | None,
) -> None:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1].")
    if not 0.0 <= soft_update_rate <= 1.0:
        raise ValueError("soft_update_rate must be in [0, 1].")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if buffer_capacity < batch_size:
        raise ValueError("buffer_capacity must be at least batch_size.")
    if learning_starts < 0:
        raise ValueError("learning_starts must be non-negative.")
    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive or None.")


@torch.no_grad()
def soft_update(
    target_net: nn.Module, online_net: nn.Module, soft_update_rate: float
) -> None:
    for target_param, online_param in zip(
        target_net.parameters(), online_net.parameters(), strict=True
    ):
        target_param.mul_(1.0 - soft_update_rate).add_(
            online_param, alpha=soft_update_rate
        )
