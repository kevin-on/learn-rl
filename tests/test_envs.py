import numpy as np
from gymnasium import spaces

from envs import (
    DiscreteActionSpec,
    EnvPoolVecEnv,
    NormalizeObservationVecEnv,
    RunningMeanStd,
    VecEnvStep,
)


class FakeDiscreteVecEnv:
    num_envs = 2
    observation_shape = (1,)
    action_spec = DiscreteActionSpec(num_actions=2)
    num_actions = 2

    def __init__(self) -> None:
        self.step_observation = np.asarray([[5.0], [7.0]], dtype=np.float32)
        self.subset_observation = np.asarray([[9.0]], dtype=np.float32)
        self.closed = False

    def reset(self) -> np.ndarray:
        return np.asarray([[1.0], [3.0]], dtype=np.float32)

    def step(self, actions: np.ndarray) -> VecEnvStep:
        assert actions.shape == (2,)
        return VecEnvStep(
            observation=self.step_observation,
            reward=np.ones(2, dtype=np.float32),
            terminated=np.zeros(2, dtype=np.bool_),
            truncated=np.zeros(2, dtype=np.bool_),
            env_id=np.asarray([0, 1], dtype=np.int32),
        )

    def reset_subset(self, env_ids: np.ndarray) -> np.ndarray:
        assert env_ids.shape == (1,)
        return self.subset_observation

    def render(self) -> np.ndarray:
        return np.zeros((2, 4, 4, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def test_running_mean_std_matches_concatenated_moments() -> None:
    first_batch = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    second_batch = np.asarray([[5.0, 6.0]], dtype=np.float32)
    all_observations = np.concatenate([first_batch, second_batch], axis=0)

    rms = RunningMeanStd(shape=(2,), epsilon=0.0)
    rms.update(first_batch)
    rms.update(second_batch)

    np.testing.assert_allclose(rms.mean, np.mean(all_observations, axis=0))
    np.testing.assert_allclose(rms.var, np.var(all_observations, axis=0))
    assert rms.count == 3.0


def test_normalize_observation_vec_env_updates_and_clips_observations() -> None:
    env = FakeDiscreteVecEnv()
    wrapper = NormalizeObservationVecEnv(
        env,
        training=True,
        observation_rms=RunningMeanStd(shape=(1,), epsilon=0.0),
        clip=1.0,
        epsilon=1e-8,
    )

    reset_observation = wrapper.reset()
    np.testing.assert_allclose(
        reset_observation,
        np.asarray([[-1.0], [1.0]], dtype=np.float32),
        rtol=1e-6,
    )
    assert wrapper.observation_rms.count == 2.0

    step = wrapper.step(np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_allclose(
        step.observation,
        np.asarray([[0.4472136], [1.0]], dtype=np.float32),
        rtol=1e-6,
    )
    assert wrapper.observation_rms.count == 4.0


def test_normalize_observation_vec_env_freezes_eval_stats() -> None:
    env = FakeDiscreteVecEnv()
    rms = RunningMeanStd(shape=(1,), epsilon=0.0)
    rms.update(np.asarray([[8.0], [12.0]], dtype=np.float32))
    wrapper = NormalizeObservationVecEnv(
        env,
        training=False,
        observation_rms=rms,
        clip=10.0,
        epsilon=1e-8,
    )

    observation = wrapper.reset()

    np.testing.assert_allclose(
        observation,
        np.asarray([[-4.5], [-3.5]], dtype=np.float32),
        rtol=1e-6,
    )
    assert rms.count == 2.0


def test_normalize_observation_vec_env_proxies_render() -> None:
    env = FakeDiscreteVecEnv()
    wrapper = NormalizeObservationVecEnv(
        env,
        training=False,
        observation_rms=RunningMeanStd(shape=(1,), epsilon=0.0),
    )

    frame = wrapper.render()

    assert frame.shape == (2, 4, 4, 3)


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


def test_envpool_vec_env_passes_render_mode(monkeypatch) -> None:
    class FakeEnv:
        action_space = spaces.Discrete(2)
        observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        def reset(self, env_ids: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
            batch_size = 2 if env_ids is None else len(env_ids)
            return np.zeros((batch_size, 2), dtype=np.float32), {}

        def step(
            self, _action: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
            return (
                np.zeros((2, 2), dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.zeros(2, dtype=np.bool_),
                np.zeros(2, dtype=np.bool_),
                {"env_id": np.asarray([0, 1], dtype=np.int32)},
            )

        def render(self) -> np.ndarray:
            return np.zeros((2, 8, 8, 3), dtype=np.uint8)

        def close(self) -> None:
            pass

    captured_kwargs = {}

    def fake_make_gymnasium(_env_id: str, **kwargs) -> FakeEnv:
        captured_kwargs.update(kwargs)
        return FakeEnv()

    monkeypatch.setattr("envs.envpool.make_gymnasium", fake_make_gymnasium)
    env = EnvPoolVecEnv(
        env_id="Fake-v0",
        num_envs=2,
        seed=123,
        render_mode="rgb_array",
    )

    try:
        assert captured_kwargs["render_mode"] == "rgb_array"
        assert env.render().shape == (2, 8, 8, 3)
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
