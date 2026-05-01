import copy
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from task_adapter import DiscreteActionTaskAdapter, State

# Replay-buffer transition: state, action, reward, next state, terminal flag.
type Experience = tuple[State, int, float, State, bool]


@dataclass(frozen=True)
class DQNLog:
    step_index: int
    loss: float | None
    reward: float
    exploration_rate: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


type DQNLogFn[Observation, EnvAction] = Callable[
    ["DQN[Observation, EnvAction]", DQNLog],
    None,
]
type DQNEvalFn[Observation, EnvAction] = Callable[
    ["DQN[Observation, EnvAction]", int, float],
    None,
]
type ExplorationRateFn = Callable[[int], float]


class DQN[Observation, EnvAction]:
    def __init__(
        self,
        task_adapter: DiscreteActionTaskAdapter[Observation, EnvAction],
        q_net: nn.Module,
        learning_rate: float,
        discount_factor: float,
        soft_update_rate: float,
        buffer_capacity: int,
    ) -> None:
        self.task_adapter = task_adapter
        self.env = task_adapter.env
        self.online_q_net = q_net
        self.target_q_net = copy.deepcopy(q_net)
        self.target_q_net.eval()
        self.device = next(self.online_q_net.parameters()).device
        self.optimizer = torch.optim.Adam(
            self.online_q_net.parameters(), lr=learning_rate
        )
        self.discount_factor = discount_factor
        self.soft_update_rate = soft_update_rate
        self.replay_buffer: deque[Experience] = deque(maxlen=buffer_capacity)

    def train(
        self,
        num_steps: int,
        batch_size: int,
        exploration_rate_fn: ExplorationRateFn,
        env_seed: int | None = None,
        log_fn: DQNLogFn[Observation, EnvAction] | None = None,
        eval_fn: DQNEvalFn[Observation, EnvAction] | None = None,
    ) -> None:
        observation, _info = self.env.reset(seed=env_seed)
        state = self.task_adapter.encode_observation(observation)
        for step_index in range(num_steps):
            exploration_rate = validate_exploration_rate(
                exploration_rate_fn(step_index)
            )
            action_index, env_action = self._select_action(state, exploration_rate)
            next_observation, reward, terminated, truncated, info = self.env.step(
                env_action
            )
            next_state = self.task_adapter.encode_observation(next_observation)
            terminal = terminated
            should_reset = terminated or truncated
            self.replay_buffer.append(
                (state, action_index, float(reward), next_state, terminal)
            )

            state = next_state
            if should_reset:
                observation, _info = self.env.reset()
                state = self.task_adapter.encode_observation(observation)

            loss_value = None
            if len(self.replay_buffer) >= batch_size:
                loss = self._compute_td_loss(batch_size)
                loss_value = float(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                soft_update(
                    self.target_q_net,
                    self.online_q_net,
                    self.soft_update_rate,
                )

            if log_fn is not None:
                log_fn(
                    self,
                    DQNLog(
                        step_index=step_index,
                        loss=loss_value,
                        reward=float(reward),
                        exploration_rate=exploration_rate,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    ),
                )
            if eval_fn is not None:
                eval_fn(self, step_index, exploration_rate)

    def _select_action(
        self, state: State, exploration_rate: float
    ) -> tuple[int, EnvAction]:
        if np.random.random() < exploration_rate:
            action_index = self.task_adapter.sample_action_index()
        else:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                q_values = self.online_q_net(state_tensor)
                self._validate_q_values(q_values)
            action_index = int(q_values.argmax(dim=1).item())

        env_action = self.task_adapter.action_index_to_env_action(action_index)
        return action_index, env_action

    def _compute_td_loss(self, batch_size: int) -> torch.Tensor:
        batch = random.sample(self.replay_buffer, batch_size)
        states, action_indices, rewards, next_states, terminals = zip(
            *batch, strict=False
        )
        states = torch.as_tensor(
            np.asarray(states), dtype=torch.float32, device=self.device
        )
        action_indices = torch.as_tensor(
            action_indices, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.asarray(next_states), dtype=torch.float32, device=self.device
        )
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

    def _validate_q_values(self, q_values: torch.Tensor) -> None:
        if q_values.ndim != 2:
            msg = f"Q-network must return a 2D tensor, got shape {q_values.shape}"
            raise ValueError(msg)

        if q_values.shape[1] != self.task_adapter.num_actions:
            msg = (
                "Q-network action dimension must match the task adapter: "
                f"expected {self.task_adapter.num_actions}, got {q_values.shape[1]}"
            )
            raise ValueError(msg)


def validate_exploration_rate(exploration_rate: float) -> float:
    exploration_rate = float(exploration_rate)
    if not 0.0 <= exploration_rate <= 1.0:
        msg = f"exploration rate must be in [0, 1], got {exploration_rate}"
        raise ValueError(msg)
    return exploration_rate


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
