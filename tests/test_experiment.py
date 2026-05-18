import numpy as np
from gymnasium import spaces

from config import (
    A2CConfig,
    A2CTrainConfig,
    AtariConfig,
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    ObservationNormalizationConfig,
    PPOConfig,
    PPOTrainConfig,
)
from envs import EnvPoolVecEnv, NormalizeObservationVecEnv
from experiment import envpool_kwargs, make_envpool_env, observation_normalization_stats


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
        return np.zeros((2, 6, 6, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


def atari_config(**overrides) -> AtariConfig:
    values = {
        "stack_num": 4,
        "frame_skip": 4,
        "noop_max": 30,
        "episodic_life": False,
        "reward_clip": False,
        "img_height": 84,
        "img_width": 84,
        "gray_scale": True,
    }
    values.update(overrides)
    return AtariConfig(**values)


def a2c_config(env: EnvConfig) -> A2CConfig:
    return A2CConfig(
        experiment=ExperimentConfig(name="test", run_root="runs"),
        seed=123,
        env=env,
        model=ModelConfig(name="discrete_mlp"),
        train=A2CTrainConfig(
            steps=100,
            learning_rate=0.001,
            value_loss_coef=0.5,
            discount_factor=0.99,
            rollout_steps=5,
            max_grad_norm=0.5,
        ),
        eval=EvalConfig(every_steps=50, episodes=1, seed=10000),
        logging=LoggingConfig(loss_every_steps=10),
    )


def ppo_config(env: EnvConfig) -> PPOConfig:
    return PPOConfig(
        experiment=ExperimentConfig(name="test", run_root="runs"),
        seed=123,
        env=env,
        model=ModelConfig(name="discrete_mlp"),
        train=PPOTrainConfig(
            steps=256,
            learning_rate=0.001,
            discount_factor=0.99,
            gae_lambda=0.95,
            rollout_steps=32,
            minibatch_size=256,
            epochs=2,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=0.5,
        ),
        eval=EvalConfig(every_steps=50, episodes=1, seed=10000),
        logging=LoggingConfig(loss_every_steps=10),
    )


def test_make_envpool_env_returns_raw_env_without_normalization(monkeypatch) -> None:
    monkeypatch.setattr(
        "envs.envpool.make_gymnasium",
        lambda _env_id, **_kwargs: FakeEnv(),
    )
    config = a2c_config(EnvConfig(id="Fake-v0", num_envs=2))

    env = make_envpool_env(config, num_envs=2, seed=1)

    try:
        assert isinstance(env, EnvPoolVecEnv)
    finally:
        env.close()


def test_make_envpool_env_wraps_and_shares_observation_stats(monkeypatch) -> None:
    monkeypatch.setattr(
        "envs.envpool.make_gymnasium",
        lambda _env_id, **_kwargs: FakeEnv(),
    )
    config = a2c_config(
        EnvConfig(
            id="Fake-v0",
            num_envs=2,
            observation_normalization=ObservationNormalizationConfig(clip=5.0),
        )
    )

    train_env = make_envpool_env(config, num_envs=2, seed=1)
    eval_env = make_envpool_env(
        config,
        num_envs=2,
        seed=2,
        evaluation=True,
        observation_rms=observation_normalization_stats(train_env),
    )

    try:
        assert isinstance(train_env, NormalizeObservationVecEnv)
        assert isinstance(eval_env, NormalizeObservationVecEnv)
        assert eval_env.observation_rms is train_env.observation_rms
        assert train_env.training is True
        assert eval_env.training is False
    finally:
        train_env.close()
        eval_env.close()


def test_normalized_env_keeps_render_capability(monkeypatch) -> None:
    captured_kwargs: list[dict] = []

    def fake_make_gymnasium(_env_id: str, **kwargs) -> FakeEnv:
        captured_kwargs.append(kwargs)
        return FakeEnv()

    monkeypatch.setattr("envs.envpool.make_gymnasium", fake_make_gymnasium)
    config = a2c_config(
        EnvConfig(
            id="Fake-v0",
            num_envs=2,
            observation_normalization=ObservationNormalizationConfig(clip=5.0),
        )
    )

    train_env = make_envpool_env(config, num_envs=2, seed=1)
    eval_env = make_envpool_env(
        config,
        num_envs=2,
        seed=2,
        evaluation=True,
        observation_rms=observation_normalization_stats(train_env),
        render_mode="rgb_array",
    )

    try:
        assert isinstance(eval_env, NormalizeObservationVecEnv)
        assert captured_kwargs[-1]["render_mode"] == "rgb_array"
        assert eval_env.render().shape == (2, 6, 6, 3)
    finally:
        train_env.close()
        eval_env.close()


def test_envpool_kwargs_uses_atari_env_config() -> None:
    config = ppo_config(
        EnvConfig(
            id="Pong-v5",
            num_envs=8,
            atari=atari_config(
                episodic_life=True,
                reward_clip=True,
            ),
        )
    )

    train_kwargs = envpool_kwargs(config)
    eval_kwargs = envpool_kwargs(config, evaluation=True)

    assert train_kwargs["stack_num"] == 4
    assert train_kwargs["episodic_life"] is True
    assert train_kwargs["reward_clip"] is True
    assert eval_kwargs["episodic_life"] is False
    assert eval_kwargs["reward_clip"] is False
