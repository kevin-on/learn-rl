from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import envpool
import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray

type ObservationBatch = NDArray[np.generic]
type RewardBatch = NDArray[np.float32]
type DoneBatch = NDArray[np.bool_]
type ActionBatch = NDArray[np.generic]
type EnvIdBatch = NDArray[np.int32]
type EnvInfoLeaf = ArrayLike | int | float | bool | str
type EnvInfo = Mapping[str, EnvInfoLeaf | Mapping[str, EnvInfoLeaf]]


@dataclass(frozen=True)
class DiscreteActionSpec:
    num_actions: int
    start: int = 0


@dataclass(frozen=True)
class BoxActionSpec:
    shape: tuple[int, ...]
    low: NDArray[np.floating[Any]]
    high: NDArray[np.floating[Any]]
    dtype: np.dtype[Any]


type ActionSpec = DiscreteActionSpec | BoxActionSpec


@dataclass(frozen=True)
class VecEnvStep:
    observation: ObservationBatch
    reward: RewardBatch
    terminated: DoneBatch
    truncated: DoneBatch
    env_id: EnvIdBatch
    info: EnvInfo = field(default_factory=dict)


class VecEnv(Protocol):
    num_envs: int
    observation_shape: tuple[int, ...]
    action_spec: ActionSpec

    def reset(self) -> ObservationBatch:
        raise NotImplementedError

    def step(self, actions: ActionBatch) -> VecEnvStep:
        raise NotImplementedError

    def reset_subset(self, env_ids: EnvIdBatch) -> ObservationBatch:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class DiscreteVecEnv(VecEnv, Protocol):
    num_actions: int


class EnvPoolVecEnv(VecEnv):
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
        if not isinstance(self.env.observation_space, spaces.Box):
            raise TypeError("EnvPoolVecEnv requires a Box observation space.")
        if isinstance(self.env.action_space, spaces.Discrete):
            self.action_spec: ActionSpec = DiscreteActionSpec(
                num_actions=int(self.env.action_space.n),
                start=int(getattr(self.env.action_space, "start", 0)),
            )
            self.num_actions = self.action_spec.num_actions
        elif isinstance(self.env.action_space, spaces.Box):
            action_space = self.env.action_space
            self.action_spec = BoxActionSpec(
                shape=tuple(int(size) for size in action_space.shape),
                low=np.asarray(action_space.low, dtype=action_space.dtype),
                high=np.asarray(action_space.high, dtype=action_space.dtype),
                dtype=np.dtype(action_space.dtype),
            )
        else:
            msg = "EnvPoolVecEnv requires a Discrete or Box action space."
            raise TypeError(msg)

        self.num_envs = num_envs
        self.observation_shape = tuple(
            int(size) for size in self.env.observation_space.shape
        )

    def reset(self) -> ObservationBatch:
        observation, _info = self.env.reset()
        return _observation_batch(observation, batch_size=self.num_envs)

    def step(self, actions: ActionBatch) -> VecEnvStep:
        env_action = self._env_action(actions)
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

    def _env_action(self, actions: ActionBatch) -> NDArray[np.generic]:
        if isinstance(self.action_spec, DiscreteActionSpec):
            action_indices = np.asarray(actions, dtype=np.int32)
            assert action_indices.shape == (self.num_envs,)
            if np.any(action_indices < 0) or np.any(
                action_indices >= self.action_spec.num_actions
            ):
                msg = f"action indices must be in [0, {self.action_spec.num_actions})."
                raise ValueError(msg)
            return action_indices + self.action_spec.start

        if isinstance(self.action_spec, BoxActionSpec):
            action_values = np.asarray(actions, dtype=self.action_spec.dtype)
            expected_shape = (self.num_envs, *self.action_spec.shape)
            assert action_values.shape == expected_shape
            return np.clip(
                action_values,
                self.action_spec.low,
                self.action_spec.high,
            ).astype(self.action_spec.dtype, copy=False)

        msg = f"Unsupported action spec: {type(self.action_spec).__name__}."
        raise TypeError(msg)


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
