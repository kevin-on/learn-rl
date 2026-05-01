from abc import ABC, abstractmethod
from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

type State = NDArray[np.float32]


class DiscreteActionTaskAdapter[Observation, EnvAction](ABC):
    def __init__(self, env: gym.Env[Observation, EnvAction]) -> None:
        self.env = env

        if not isinstance(env.action_space, spaces.Discrete):
            msg = "DiscreteActionTaskAdapter requires a Gym Discrete action space."
            raise TypeError(msg)

    @property
    def num_actions(self) -> int:
        return int(self._action_space.n)

    @abstractmethod
    def encode_observation(self, observation: Observation) -> State:
        raise NotImplementedError

    def sample_action_index(self) -> int:
        return int(np.random.randint(self.num_actions))

    def action_index_to_env_action(self, action_index: int) -> EnvAction:
        self._validate_action_index(action_index)
        return cast(EnvAction, int(self._action_space.start) + action_index)

    def _validate_action_index(self, action_index: int) -> None:
        if not 0 <= action_index < self.num_actions:
            msg = f"action index {action_index} is outside [0, {self.num_actions})"
            raise ValueError(msg)

    @property
    def _action_space(self) -> spaces.Discrete:
        return cast(spaces.Discrete, self.env.action_space)


class CartPoleTaskAdapter(DiscreteActionTaskAdapter[NDArray[np.float32], int]):
    def encode_observation(self, observation: NDArray[np.float32]) -> State:
        state = np.asarray(observation, dtype=np.float32)
        if state.shape != (4,):
            msg = f"CartPole observation must have shape (4,), got {state.shape}"
            raise ValueError(msg)
        return state
