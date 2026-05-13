from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import envpool
import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray

type ObservationBatch = NDArray[np.generic]
type RewardBatch = NDArray[np.float32]
type DoneBatch = NDArray[np.bool_]
type ActionBatch = NDArray[np.int32]
type EnvIdBatch = NDArray[np.int32]
type EnvInfoLeaf = ArrayLike | int | float | bool | str
type EnvInfo = Mapping[str, EnvInfoLeaf | Mapping[str, EnvInfoLeaf]]


@dataclass(frozen=True)
class VecEnvStep:
    observation: ObservationBatch
    reward: RewardBatch
    terminated: DoneBatch
    truncated: DoneBatch
    env_id: EnvIdBatch
    info: EnvInfo = field(default_factory=dict)


class DiscreteVecEnv(Protocol):
    num_envs: int
    num_actions: int
    observation_shape: tuple[int, ...]

    def reset(self) -> ObservationBatch:
        raise NotImplementedError

    def step(self, action_indices: ActionBatch) -> VecEnvStep:
        raise NotImplementedError

    def reset_subset(self, env_ids: EnvIdBatch) -> ObservationBatch:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class EnvPoolVecEnv(DiscreteVecEnv):
    def __init__(
        self,
        *,
        env_id: str,
        num_envs: int,
        seed: int,
        env_kwargs: Mapping[str, int | bool] | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")

        kwargs: dict[str, int | bool] = {
            "num_envs": num_envs,
            "batch_size": num_envs,
            "seed": seed,
        }
        kwargs.update(env_kwargs or {})
        self.env = envpool.make_gymnasium(env_id, **kwargs)
        if not isinstance(self.env.action_space, spaces.Discrete):
            raise TypeError("EnvPoolVecEnv requires a discrete action space.")
        if not isinstance(self.env.observation_space, spaces.Box):
            raise TypeError("EnvPoolVecEnv requires a Box observation space.")

        self.num_envs = num_envs
        self.num_actions = int(self.env.action_space.n)
        self.observation_shape = tuple(
            int(size) for size in self.env.observation_space.shape
        )
        self._action_start = int(getattr(self.env.action_space, "start", 0))

    def reset(self) -> ObservationBatch:
        observation, _info = self.env.reset()
        return _observation_batch(observation, batch_size=self.num_envs)

    def step(self, action_indices: ActionBatch) -> VecEnvStep:
        action_indices = np.asarray(action_indices, dtype=np.int32)
        assert action_indices.shape == (self.num_envs,)
        if np.any(action_indices < 0) or np.any(action_indices >= self.num_actions):
            msg = f"action indices must be in [0, {self.num_actions})."
            raise ValueError(msg)

        env_action = action_indices + self._action_start
        observation, reward, terminated, truncated, info = self.env.step(env_action)

        return VecEnvStep(
            observation=_observation_batch(observation, batch_size=self.num_envs),
            reward=_vector_array(reward, dtype=np.float32, batch_size=self.num_envs),
            terminated=_vector_array(
                terminated, dtype=np.bool_, batch_size=self.num_envs
            ),
            truncated=_vector_array(
                truncated, dtype=np.bool_, batch_size=self.num_envs
            ),
            env_id=_env_ids_from_info(info=info, batch_size=self.num_envs),
            info=info,
        )

    def reset_subset(self, env_ids: EnvIdBatch) -> ObservationBatch:
        env_ids = np.asarray(env_ids, dtype=np.int32)
        assert env_ids.ndim == 1
        observation, _info = self.env.reset(env_ids)
        return _observation_batch(observation, batch_size=len(env_ids))

    def close(self) -> None:
        self.env.close()


def _observation_batch(value: ArrayLike, *, batch_size: int) -> ObservationBatch:
    observation = np.array(value, copy=True)
    assert observation.shape[0] == batch_size
    return observation


def _vector_array(
    value: ArrayLike,
    *,
    dtype: type[np.generic],
    batch_size: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    assert array.shape == (batch_size,)
    return array


def _env_ids_from_info(*, info: EnvInfo, batch_size: int) -> EnvIdBatch:
    raw_env_ids = info.get("env_id", np.arange(batch_size, dtype=np.int32))
    env_ids = np.asarray(raw_env_ids, dtype=np.int32)
    assert env_ids.shape == (batch_size,)
    return env_ids
