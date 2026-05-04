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

    @property
    @abstractmethod
    def state_size(self) -> int:
        raise NotImplementedError

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


class VectorObservationTaskAdapter(DiscreteActionTaskAdapter[NDArray[np.float32], int]):
    def __init__(
        self,
        env: gym.Env[NDArray[np.float32], int],
        *,
        observation_shape: tuple[int, ...],
        task_name: str,
    ) -> None:
        super().__init__(env)
        self.observation_shape = observation_shape
        self.task_name = task_name

        if not isinstance(env.observation_space, spaces.Box):
            msg = f"{task_name} requires a Gym Box observation space."
            raise TypeError(msg)
        if env.observation_space.shape != observation_shape:
            msg = (
                f"{task_name} observation space must have shape "
                f"{observation_shape}, got {env.observation_space.shape}"
            )
            raise ValueError(msg)

    @property
    def state_size(self) -> int:
        return int(np.prod(self.observation_shape))

    def encode_observation(self, observation: NDArray[np.float32]) -> State:
        state = np.asarray(observation, dtype=np.float32)
        if state.shape != self.observation_shape:
            msg = (
                f"{self.task_name} observation must have shape "
                f"{self.observation_shape}, got {state.shape}"
            )
            raise ValueError(msg)
        return state.reshape(self.state_size)


class CartPoleTaskAdapter(VectorObservationTaskAdapter):
    def __init__(self, env: gym.Env[NDArray[np.float32], int]) -> None:
        super().__init__(env, observation_shape=(4,), task_name="CartPole")


class MountainCarTaskAdapter(VectorObservationTaskAdapter):
    def __init__(self, env: gym.Env[NDArray[np.float32], int]) -> None:
        super().__init__(env, observation_shape=(2,), task_name="MountainCar")


class AcrobotTaskAdapter(VectorObservationTaskAdapter):
    def __init__(self, env: gym.Env[NDArray[np.float32], int]) -> None:
        super().__init__(env, observation_shape=(6,), task_name="Acrobot")


class LunarLanderTaskAdapter(VectorObservationTaskAdapter):
    def __init__(self, env: gym.Env[NDArray[np.float32], int]) -> None:
        super().__init__(env, observation_shape=(8,), task_name="LunarLander")


type VectorTaskAdapter = DiscreteActionTaskAdapter[NDArray[np.float32], int]


def make_task_adapter(
    env: gym.Env[NDArray[np.float32], int], env_id: str
) -> VectorTaskAdapter:
    if env_id == "CartPole-v1":
        return CartPoleTaskAdapter(env)
    if env_id == "MountainCar-v0":
        return MountainCarTaskAdapter(env)
    if env_id == "Acrobot-v1":
        return AcrobotTaskAdapter(env)
    if env_id == "LunarLander-v3":
        return LunarLanderTaskAdapter(env)

    supported_envs = ", ".join(
        ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "LunarLander-v3"]
    )
    msg = f"Unsupported env.id {env_id!r}. Supported environments: {supported_envs}."
    raise ValueError(msg)
