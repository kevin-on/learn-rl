import copy
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ddpg import (
    NormalActionNoise,
    soft_update,
)
from envs import BoxActionSpec, ObservationBatch, VecEnv
from experiment import model_device
from training_state import EpisodeTracker, ReplayBuffer

type State = np.ndarray
type Action = np.ndarray
type Experience = tuple[State, Action, float, State, bool]


@dataclass(frozen=True)
class TD3Episode:
    env_id: int
    episode_return: float
    episode_length: int


@dataclass(frozen=True)
class TD3UpdateStats:
    actor_loss: float | None
    critic_loss: float
    critic1_loss: float
    critic2_loss: float
    q1_mean: float
    q2_mean: float
    target_q_mean: float


@dataclass(frozen=True)
class TD3Log:
    step: int
    update: int
    stats: TD3UpdateStats | None
    reward_mean: float
    episodes: tuple[TD3Episode, ...] = ()


type TD3LogFn = Callable[["TD3", TD3Log], None]


class TD3:
    def __init__(
        self,
        env: VecEnv,
        model: nn.Module,
        *,
        actor_learning_rate: float,
        critic_learning_rate: float,
        discount_factor: float,
        soft_update_rate: float,
        buffer_capacity: int,
        batch_size: int,
        learning_starts: int = 0,
        exploration_sigma: float = 0.1,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
    ) -> None:
        validate_td3_hyperparameters(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            discount_factor=discount_factor,
            soft_update_rate=soft_update_rate,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            learning_starts=learning_starts,
            exploration_sigma=exploration_sigma,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            policy_delay=policy_delay,
        )
        if not isinstance(env.action_spec, BoxActionSpec):
            raise ValueError("TD3 requires a Box action space.")
        if not all(hasattr(model, name) for name in ("actor", "critic1", "critic2")):
            raise TypeError("TD3 model must expose actor, critic1, and critic2.")
        if not all(callable(getattr(model, name, None)) for name in ("act", "q1")):
            raise TypeError("TD3 model must expose act() and q1() methods.")
        if not callable(getattr(model, "q_pair", None)):
            raise TypeError("TD3 model must expose q_pair() method.")

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
            itertools.chain(
                self.online_model.critic1.parameters(),
                self.online_model.critic2.parameters(),
            ),
            lr=critic_learning_rate,
        )
        self.num_envs = env.num_envs
        self.action_spec = env.action_spec
        self.discount_factor = discount_factor
        self.soft_update_rate = soft_update_rate
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay
        self.replay_buffer = ReplayBuffer[Experience](buffer_capacity)
        self.step = 0
        self.update = 0
        self._episode_tracker = EpisodeTracker(self.num_envs)
        self._action_low = np.asarray(self.action_spec.low, dtype=np.float32)
        self._action_high = np.asarray(self.action_spec.high, dtype=np.float32)
        self._action_scale = (self._action_high - self._action_low) / 2.0
        self._action_low_tensor = torch.as_tensor(
            self._action_low,
            dtype=torch.float32,
            device=self.device,
        )
        self._action_high_tensor = torch.as_tensor(
            self._action_high,
            dtype=torch.float32,
            device=self.device,
        )
        self._action_scale_tensor = torch.as_tensor(
            self._action_scale,
            dtype=torch.float32,
            device=self.device,
        )
        noise_shape = (self.num_envs, *self.action_spec.shape)
        self._action_noise = NormalActionNoise(
            shape=noise_shape,
            sigma=exploration_sigma,
        )

    def train(
        self,
        num_steps: int,
        log_fn: TD3LogFn | None = None,
    ) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        observation = self.env.reset()
        self._action_noise.reset()
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
                self._action_noise.reset(done_slots)
                reset_observation = self.env.reset_subset(env_step.env_id[done_slots])
                assert reset_observation.shape == (
                    len(done_slots),
                    *self.env.observation_shape,
                )
                next_observation[done_slots] = reset_observation
            observation = next_observation

            stats = None
            if (
                self.step >= self.learning_starts
                and len(self.replay_buffer) >= self.batch_size
            ):
                stats = self._update()
                self.update += 1

            self.step += self.num_envs
            if log_fn is not None:
                log_fn(
                    self,
                    TD3Log(
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
                "replay_buffer": self.replay_buffer.checkpoint_state(),
            },
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self.online_model.load_state_dict(state["model_state"])
        optimizer_state = state["optimizer_state"]
        self.actor_optimizer.load_state_dict(optimizer_state["actor"])
        self.critic_optimizer.load_state_dict(optimizer_state["critic"])
        algorithm_state = state["algorithm_state"]
        self.target_model.load_state_dict(algorithm_state["target_model_state"])
        self.replay_buffer.load_checkpoint_state(algorithm_state["replay_buffer"])
        self.step = int(state["step"])
        self.update = int(state["update"])
        self._episode_tracker.reset()
        self._action_noise.reset()

    @torch.no_grad()
    def _select_actions(
        self,
        observation: ObservationBatch,
    ) -> np.ndarray:
        if self.step < self.learning_starts:
            actions = np.random.uniform(
                low=self._action_low,
                high=self._action_high,
                size=(self.num_envs, *self.action_spec.shape),
            )
            return actions.astype(self.action_spec.dtype, copy=False)

        observation_tensor = torch.as_tensor(observation, device=self.device)
        actions = self.online_model.act(observation_tensor).cpu().numpy()
        actions = actions + self._action_noise.sample() * self._action_scale
        return np.clip(actions, self._action_low, self._action_high).astype(
            self.action_spec.dtype,
            copy=False,
        )

    def _update(self) -> TD3UpdateStats:
        batch = self.replay_buffer.sample(self.batch_size)
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
            target_noise = (
                torch.randn_like(next_actions)
                * self.target_policy_noise
                * self._action_scale_tensor
            )
            target_noise_limit = self.target_noise_clip * self._action_scale_tensor
            target_noise = torch.clamp(
                target_noise,
                min=-target_noise_limit,
                max=target_noise_limit,
            )
            next_actions = torch.clamp(
                next_actions + target_noise,
                min=self._action_low_tensor,
                max=self._action_high_tensor,
            )
            next_q1_values, next_q2_values = self.target_model.q_pair(
                next_states,
                next_actions,
            )
            next_q_values = torch.minimum(next_q1_values, next_q2_values)
            target_q_values = (
                rewards + self.discount_factor * (1.0 - terminals) * next_q_values
            )

        q1_values, q2_values = self.online_model.q_pair(states, actions)
        critic1_loss = F.mse_loss(q1_values, target_q_values)
        critic2_loss = F.mse_loss(q2_values, target_q_values)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss: torch.Tensor | None = None
        if (self.update + 1) % self.policy_delay == 0:
            policy_actions = self.online_model.act(states)
            actor_loss = -self.online_model.q1(states, policy_actions).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(
                self.target_model,
                self.online_model,
                self.soft_update_rate,
            )

        return TD3UpdateStats(
            actor_loss=None if actor_loss is None else float(actor_loss.item()),
            critic_loss=float(critic_loss.item()),
            critic1_loss=float(critic1_loss.item()),
            critic2_loss=float(critic2_loss.item()),
            q1_mean=float(q1_values.detach().mean().item()),
            q2_mean=float(q2_values.detach().mean().item()),
            target_q_mean=float(target_q_values.detach().mean().item()),
        )

    def _record_episodes(
        self,
        rewards: np.ndarray,
        done: np.ndarray,
        env_ids: np.ndarray,
    ) -> list[TD3Episode]:
        return self._episode_tracker.record(
            rewards=rewards,
            done=done,
            env_ids=env_ids,
            episode_factory=TD3Episode,
        )


def validate_td3_hyperparameters(
    *,
    actor_learning_rate: float,
    critic_learning_rate: float,
    discount_factor: float,
    soft_update_rate: float,
    buffer_capacity: int,
    batch_size: int,
    learning_starts: int,
    exploration_sigma: float,
    target_policy_noise: float,
    target_noise_clip: float,
    policy_delay: int,
) -> None:
    if actor_learning_rate <= 0.0:
        raise ValueError("actor_learning_rate must be positive.")
    if critic_learning_rate <= 0.0:
        raise ValueError("critic_learning_rate must be positive.")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be in [0, 1].")
    if not 0.0 <= soft_update_rate <= 1.0:
        raise ValueError("soft_update_rate must be in [0, 1].")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if learning_starts < 0:
        raise ValueError("learning_starts must be non-negative.")
    if buffer_capacity < batch_size:
        raise ValueError("buffer_capacity must be at least batch_size.")
    if exploration_sigma < 0.0:
        raise ValueError("exploration_sigma must be non-negative.")
    if target_policy_noise < 0.0:
        raise ValueError("target_policy_noise must be non-negative.")
    if target_noise_clip < 0.0:
        raise ValueError("target_noise_clip must be non-negative.")
    if policy_delay <= 0:
        raise ValueError("policy_delay must be positive.")
