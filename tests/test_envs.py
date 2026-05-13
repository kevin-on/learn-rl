import numpy as np
from gymnasium import spaces

from envs import EnvPoolVecEnv


def test_envpool_vec_env_cartpole_shapes() -> None:
    env = EnvPoolVecEnv(env_id="CartPole-v1", num_envs=2, seed=123)
    try:
        observation = env.reset()
        assert observation.shape == (2, 4)

        step = env.step(np.asarray([0, 1], dtype=np.int32))
        assert step.observation.shape == (2, 4)
        assert step.reward.shape == (2,)
        assert step.terminated.shape == (2,)
        assert step.truncated.shape == (2,)
        assert step.env_id.shape == (2,)

        reset_observation = env.reset_subset(np.asarray([0], dtype=np.int32))
        assert reset_observation.shape == (1, 4)
    finally:
        env.close()


def test_envpool_vec_env_applies_discrete_action_offset(monkeypatch) -> None:
    class FakeEnv:
        action_space = spaces.Discrete(3, start=1)
        observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        def __init__(self) -> None:
            self.last_action: np.ndarray | None = None

        def reset(self, env_ids: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
            batch_size = 2 if env_ids is None else len(env_ids)
            return np.zeros((batch_size, 2), dtype=np.float32), {}

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
            self.last_action = np.asarray(action, dtype=np.int32)
            return (
                np.zeros((2, 2), dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.zeros(2, dtype=np.bool_),
                np.zeros(2, dtype=np.bool_),
                {"env_id": np.asarray([0, 1], dtype=np.int32)},
            )

        def close(self) -> None:
            pass

    fake_env = FakeEnv()

    def fake_make_gymnasium(_env_id: str, **_kwargs: int | bool) -> FakeEnv:
        return fake_env

    monkeypatch.setattr("envs.envpool.make_gymnasium", fake_make_gymnasium)
    env = EnvPoolVecEnv(env_id="Fake-v0", num_envs=2, seed=123)
    env.step(np.asarray([0, 2], dtype=np.int32))
    assert fake_env.last_action is not None
    np.testing.assert_array_equal(fake_env.last_action, np.asarray([1, 3]))


def test_envpool_vec_env_accepts_and_clips_box_actions(monkeypatch) -> None:
    class FakeEnv:
        action_space = spaces.Box(
            low=np.asarray([-1.0, -2.0], dtype=np.float32),
            high=np.asarray([1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        def __init__(self) -> None:
            self.last_action: np.ndarray | None = None

        def reset(self, env_ids: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
            batch_size = 2 if env_ids is None else len(env_ids)
            return np.zeros((batch_size, 2), dtype=np.float32), {}

        def step(
            self, action: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
            self.last_action = np.asarray(action, dtype=np.float32)
            return (
                np.zeros((2, 2), dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.zeros(2, dtype=np.bool_),
                np.zeros(2, dtype=np.bool_),
                {"env_id": np.asarray([0, 1], dtype=np.int32)},
            )

        def close(self) -> None:
            pass

    fake_env = FakeEnv()

    def fake_make_gymnasium(_env_id: str, **_kwargs: int | bool) -> FakeEnv:
        return fake_env

    monkeypatch.setattr("envs.envpool.make_gymnasium", fake_make_gymnasium)
    env = EnvPoolVecEnv(env_id="FakeContinuous-v0", num_envs=2, seed=123)
    env.step(
        np.asarray(
            [
                [-2.0, -3.0],
                [0.5, 3.0],
            ],
            dtype=np.float32,
        )
    )
    assert fake_env.last_action is not None
    np.testing.assert_allclose(
        fake_env.last_action,
        np.asarray(
            [
                [-1.0, -2.0],
                [0.5, 2.0],
            ],
            dtype=np.float32,
        ),
    )
