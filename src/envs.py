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
    @property
    def num_envs(self) -> int:
        raise NotImplementedError

    @property
    def observation_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def action_spec(self) -> ActionSpec:
        raise NotImplementedError

    def reset(self) -> ObservationBatch:
        raise NotImplementedError

    def step(self, actions: ActionBatch) -> VecEnvStep:
        raise NotImplementedError

    def reset_subset(self, env_ids: EnvIdBatch) -> ObservationBatch:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class DiscreteVecEnv(VecEnv, Protocol):
    @property
    def num_actions(self) -> int:
        raise NotImplementedError


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
            self._action_spec: ActionSpec = DiscreteActionSpec(
                num_actions=int(self.env.action_space.n),
                start=int(getattr(self.env.action_space, "start", 0)),
            )
        elif isinstance(self.env.action_space, spaces.Box):
            action_space = self.env.action_space
            self._action_spec = BoxActionSpec(
                shape=tuple(int(size) for size in action_space.shape),
                low=np.asarray(action_space.low, dtype=action_space.dtype),
                high=np.asarray(action_space.high, dtype=action_space.dtype),
                dtype=np.dtype(action_space.dtype),
            )
        else:
            msg = "EnvPoolVecEnv requires a Discrete or Box action space."
            raise TypeError(msg)

        self._num_envs = num_envs
        self._observation_shape = tuple(
            int(size) for size in self.env.observation_space.shape
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self._observation_shape

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    @property
    def num_actions(self) -> int:
        action_spec = self.action_spec
        if not isinstance(action_spec, DiscreteActionSpec):
            raise AttributeError("continuous-action envs do not have num_actions.")
        return action_spec.num_actions

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


class RunningMeanStd:
    def __init__(self, *, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative.")

        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, batch: ObservationBatch) -> None:
        batch_array = np.asarray(batch, dtype=np.float64)
        if batch_array.ndim == 0:
            raise ValueError("batch must include a leading batch dimension.")
        if batch_array.shape[1:] != self.mean.shape:
            msg = (
                "batch observation shape must match running-stat shape: "
                f"{batch_array.shape[1:]} != {self.mean.shape}."
            )
            raise ValueError(msg)
        if batch_array.shape[0] == 0:
            return

        self.update_from_moments(
            batch_mean=np.mean(batch_array, axis=0),
            batch_var=np.var(batch_array, axis=0),
            batch_count=batch_array.shape[0],
        )

    def update_from_moments(
        self,
        *,
        batch_mean: NDArray[np.float64],
        batch_var: NDArray[np.float64],
        batch_count: int,
    ) -> None:
        if batch_count < 0:
            raise ValueError("batch_count must be non-negative.")
        if batch_count == 0:
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        current_m2 = self.var * self.count
        batch_m2 = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        new_var = (current_m2 + batch_m2 + correction) / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 0.0)
        self.count = total_count


class NormalizeObservationVecEnv(VecEnv):
    def __init__(
        self,
        env: VecEnv,
        *,
        training: bool,
        observation_rms: RunningMeanStd | None = None,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        if clip <= 0.0:
            raise ValueError("clip must be positive.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        self.env = env
        self.training = training
        self.observation_rms = observation_rms or RunningMeanStd(
            shape=self.observation_shape
        )
        if self.observation_rms.mean.shape != self.observation_shape:
            msg = (
                "observation_rms shape must match env.observation_shape: "
                f"{self.observation_rms.mean.shape} != {self.observation_shape}."
            )
            raise ValueError(msg)
        self.clip = float(clip)
        self.epsilon = float(epsilon)

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return self.env.observation_shape

    @property
    def action_spec(self) -> ActionSpec:
        return self.env.action_spec

    @property
    def num_actions(self) -> int:
        action_spec = self.action_spec
        if not isinstance(action_spec, DiscreteActionSpec):
            raise AttributeError("continuous-action envs do not have num_actions.")
        return action_spec.num_actions

    def reset(self) -> ObservationBatch:
        observation = self.env.reset()
        return self._update_and_normalize(observation)

    def step(self, actions: ActionBatch) -> VecEnvStep:
        step = self.env.step(actions)
        return VecEnvStep(
            observation=self._update_and_normalize(step.observation),
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
            env_id=step.env_id,
            info=step.info,
        )

    def reset_subset(self, env_ids: EnvIdBatch) -> ObservationBatch:
        observation = self.env.reset_subset(env_ids)
        return self._update_and_normalize(observation)

    def normalize_observation(self, observation: ObservationBatch) -> ObservationBatch:
        normalized = (
            np.asarray(observation, dtype=np.float64) - self.observation_rms.mean
        ) / np.sqrt(self.observation_rms.var + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def close(self) -> None:
        self.env.close()

    def _update_and_normalize(self, observation: ObservationBatch) -> ObservationBatch:
        if self.training:
            self.observation_rms.update(observation)
        return self.normalize_observation(observation)


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
