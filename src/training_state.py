import random
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

import numpy as np

EpisodeT = TypeVar("EpisodeT")
ExperienceT = TypeVar("ExperienceT")


class EpisodeTracker:
    def __init__(self, num_envs: int) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")

        self._returns = np.zeros(num_envs, dtype=np.float64)
        self._lengths = np.zeros(num_envs, dtype=np.int64)

    def record(
        self,
        *,
        rewards: np.ndarray,
        done: np.ndarray,
        env_ids: np.ndarray,
        episode_factory: Callable[[int, float, int], EpisodeT],
    ) -> list[EpisodeT]:
        if rewards.shape != self._returns.shape:
            msg = f"rewards shape must be {self._returns.shape}, got {rewards.shape}."
            raise ValueError(msg)
        if done.shape != self._returns.shape:
            msg = f"done shape must be {self._returns.shape}, got {done.shape}."
            raise ValueError(msg)
        if env_ids.shape != self._returns.shape:
            msg = f"env_ids shape must be {self._returns.shape}, got {env_ids.shape}."
            raise ValueError(msg)

        self._returns += rewards.astype(np.float64)
        self._lengths += 1

        episodes: list[EpisodeT] = []
        for env_slot in np.flatnonzero(done):
            episodes.append(
                episode_factory(
                    int(env_ids[env_slot]),
                    float(self._returns[env_slot]),
                    int(self._lengths[env_slot]),
                )
            )
            self._returns[env_slot] = 0.0
            self._lengths[env_slot] = 0
        return episodes

    def reset(self) -> None:
        self._returns.fill(0.0)
        self._lengths.fill(0)


class ReplayBuffer[ExperienceT]:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self._items: deque[ExperienceT] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        maxlen = self._items.maxlen
        if maxlen is None:
            raise RuntimeError("replay buffer capacity is unexpectedly unbounded.")
        return maxlen

    def append(self, experience: ExperienceT) -> None:
        self._items.append(experience)

    def sample(self, batch_size: int) -> list[ExperienceT]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > len(self._items):
            raise ValueError("batch_size must be at most replay buffer length.")
        return random.sample(self._items, batch_size)

    def checkpoint_state(self) -> list[ExperienceT]:
        return list(self._items)

    def load_checkpoint_state(self, experiences: Iterable[ExperienceT]) -> None:
        self._items = deque(experiences, maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ExperienceT]:
        return iter(self._items)
