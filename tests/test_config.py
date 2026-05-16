from pathlib import Path

import pytest

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
    load_a2c_config,
    load_a3c_config,
    load_config,
    load_ddpg_config,
    load_ppo_config,
    save_config,
)


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def a2c_yaml(env_yaml: str, train_yaml: str | None = None) -> str:
    train = (
        train_yaml
        or """
train:
  steps: 100
  learning_rate: 0.001
  value_loss_coef: 0.5
  discount_factor: 0.99
  rollout_steps: 5
  max_grad_norm: 0.5
"""
    )
    return f"""
experiment:
  name: test
  run_root: runs

seed: 123

{env_yaml}

model:
  name: discrete_mlp

{train}

eval:
  every_steps: 50
  episodes: 1
  seed: 10000

logging:
  loss_every_steps: 10
"""


def dqn_yaml(env_yaml: str) -> str:
    return f"""
experiment:
  name: test
  run_root: runs

seed: 123

{env_yaml}

model:
  name: mlp

train:
  steps: 100
  batch_size: 32
  buffer_capacity: 100
  learning_starts: 0
  learning_rate: 0.001
  discount_factor: 0.99
  soft_update_rate: 0.005
  max_grad_norm: 10.0
  exploration:
    schedule: linear
    start: 1.0
    end: 0.05
    decay_steps: 100

eval:
  every_steps: 50
  episodes: 1
  seed: 10000

logging:
  loss_every_steps: 10
"""


def ddpg_yaml(env_yaml: str) -> str:
    return f"""
experiment:
  name: test
  run_root: runs

seed: 123

{env_yaml}

model:
  name: ddpg_mlp

train:
  steps: 100
  batch_size: 32
  buffer_capacity: 100
  actor_learning_rate: 0.0001
  critic_learning_rate: 0.001
  critic_weight_decay: 0.01
  discount_factor: 0.99
  soft_update_rate: 0.001
  exploration: {{}}

eval:
  every_steps: 50
  episodes: 1
  seed: 10000

logging:
  loss_every_steps: 10
"""


def a3c_yaml(env_yaml: str) -> str:
    return f"""
experiment:
  name: test
  run_root: runs

seed: 123

{env_yaml}

model:
  name: discrete_mlp

train:
  steps: 100
  num_workers: 2
  learning_rate: 0.001
  value_loss_coef: 0.5
  discount_factor: 0.99
  rollout_steps: 5
  max_grad_norm: 0.5
  entropy_coef: 0.01
  rmsprop_alpha: 0.99
  rmsprop_eps: 0.00001

eval:
  every_steps: 50
  episodes: 1
  seed: 10000

logging:
  loss_every_steps: 10
"""


def ppo_yaml(env_yaml: str) -> str:
    return f"""
experiment:
  name: test
  run_root: runs

seed: 123

{env_yaml}

model:
  name: discrete_mlp

train:
  steps: 256
  learning_rate: 0.001
  discount_factor: 0.99
  gae_lambda: 0.95
  rollout_steps: 32
  minibatch_size: 256
  epochs: 2
  clip_coef: 0.2
  value_coef: 0.5
  entropy_coef: 0.0
  max_grad_norm: 0.5

eval:
  every_steps: 50
  episodes: 1
  seed: 10000

logging:
  loss_every_steps: 10
"""


def atari_yaml() -> str:
    return """
  atari:
    stack_num: 4
    frame_skip: 4
    noop_max: 30
    episodic_life: false
    reward_clip: false
    img_height: 84
    img_width: 84
    gray_scale: true
"""


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


def make_a2c_config(env: EnvConfig) -> A2CConfig:
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


def test_load_a2c_config_parses_base_env_without_atari(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        a2c_yaml(
            """
env:
  id: CartPole-v1
  num_envs: 8
"""
        ),
    )

    config = load_a2c_config(path)

    assert isinstance(config.env, EnvConfig)
    assert config.env.id == "CartPole-v1"
    assert config.env.num_envs == 8
    assert config.env.atari is None
    assert config.env.observation_normalization is None
    assert config.model.kwargs == {}
    assert config.checkpoint.every_steps is None


def test_load_a2c_config_parses_checkpoint_config(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        a2c_yaml(
            """
env:
  id: CartPole-v1
  num_envs: 8
"""
        )
        + """
checkpoint:
  every_steps: 50
""",
    )

    config = load_a2c_config(path)

    assert config.checkpoint.every_steps == 50
    assert config.logging.save_plot is True


def test_load_a2c_config_requires_num_envs(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        a2c_yaml(
            """
env:
  id: CartPole-v1
"""
        ),
    )

    with pytest.raises(ValueError, match=r"env\.num_envs"):
        load_a2c_config(path)


def test_load_a2c_config_requires_train_fields(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        a2c_yaml(
            """
env:
  id: CartPole-v1
  num_envs: 8
""",
            train_yaml="""
train:
  learning_rate: 0.001
  value_loss_coef: 0.5
  discount_factor: 0.99
  rollout_steps: 5
  max_grad_norm: 0.5
""",
        ),
    )

    with pytest.raises(ValueError, match=r"train\.steps"):
        load_a2c_config(path)


def test_load_ppo_config_parses_nested_atari_config(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        ppo_yaml(
            f"""
env:
  id: Pong-v5
  num_envs: 8
{atari_yaml()}
"""
        ),
    )

    config = load_ppo_config(path)

    assert isinstance(config.env, EnvConfig)
    assert config.env.atari == atari_config()


def test_load_ppo_config_accepts_nested_atari_override(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        ppo_yaml(
            f"""
env:
  id: Pong-v5
  num_envs: 8
{atari_yaml()}
"""
        ),
    )

    config = load_ppo_config(
        path,
        overrides=[
            "env.atari.frame_skip=2",
            "env.atari.reward_clip=true",
        ],
    )

    assert config.env.atari == atari_config(frame_skip=2, reward_clip=True)


def test_load_ppo_config_rejects_incomplete_atari_config(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        ppo_yaml(
            """
env:
  id: Pong-v5
  num_envs: 8
  atari:
    stack_num: 4
"""
        ),
    )

    with pytest.raises(ValueError, match=r"env\.atari\.frame_skip"):
        load_ppo_config(path)


def test_observation_normalization_uses_nested_defaults(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        a2c_yaml(
            """
env:
  id: Pendulum-v1
  num_envs: 8
  observation_normalization: {}
"""
        ),
    )

    config = load_a2c_config(path)

    assert config.env.observation_normalization == ObservationNormalizationConfig()


def test_observation_normalization_accepts_empty_override(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        a2c_yaml(
            """
env:
  id: Pendulum-v1
  num_envs: 8
"""
        ),
    )

    config = load_a2c_config(path, overrides=["env.observation_normalization={}"])

    assert config.env.observation_normalization == ObservationNormalizationConfig()


def test_observation_normalization_accepts_nested_override(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        a2c_yaml(
            """
env:
  id: Pendulum-v1
  num_envs: 8
"""
        ),
    )

    config = load_a2c_config(
        path,
        overrides=[
            "env.observation_normalization.clip=5.0",
            "env.observation_normalization.epsilon=0.000001",
        ],
    )

    assert config.env.observation_normalization == ObservationNormalizationConfig(
        clip=5.0,
        epsilon=0.000001,
    )


def test_save_config_omits_none_env_blocks_for_nested_overrides(tmp_path: Path) -> None:
    config = make_a2c_config(EnvConfig(id="Pendulum-v1", num_envs=8))
    path = tmp_path / "saved.yaml"

    save_config(config, path)

    saved_yaml = path.read_text(encoding="utf-8")
    assert "atari:" not in saved_yaml
    assert "observation_normalization:" not in saved_yaml

    loaded = load_a2c_config(
        path,
        overrides=["env.observation_normalization.clip=5.0"],
    )
    assert loaded.env.observation_normalization == ObservationNormalizationConfig(
        clip=5.0,
    )


def test_load_ppo_config_rejects_old_type_key(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        ppo_yaml(
            """
env:
  type: vector
  id: CartPole-v1
  num_envs: 8
"""
        ),
    )

    with pytest.raises(ValueError, match=r"env\.type"):
        load_ppo_config(path)


def test_load_ppo_config_rejects_old_kind_key(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        ppo_yaml(
            """
env:
  id: CartPole-v1
  kind: vector
  num_envs: 8
"""
        ),
    )

    with pytest.raises(ValueError, match=r"env\.kind"):
        load_ppo_config(path)


def test_load_a2c_config_rejects_old_flat_normalization_key(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        a2c_yaml(
            """
env:
  id: CartPole-v1
  num_envs: 8
  normalize_observation: true
"""
        ),
    )

    with pytest.raises(ValueError, match=r"env\.normalize_observation"):
        load_a2c_config(path)


def test_dqn_rejects_observation_normalization(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        dqn_yaml(
            """
env:
  id: CartPole-v1
  num_envs: 1
  observation_normalization: {}
"""
        ),
    )

    with pytest.raises(ValueError, match="not supported for DQN"):
        load_config(path)


def test_load_ddpg_config_parses_defaults(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        ddpg_yaml(
            """
env:
  id: Pendulum-v1
  num_envs: 4
"""
        ),
    )

    config = load_ddpg_config(path)

    assert config.env.id == "Pendulum-v1"
    assert config.model.name == "ddpg_mlp"
    assert config.train.learning_starts == 0
    assert config.train.actor_learning_rate == 0.0001
    assert config.train.critic_learning_rate == 0.001
    assert config.train.exploration.noise_type == "ornstein-uhlenbeck"
    assert config.train.exploration.theta == 0.15
    assert config.train.exploration.sigma == 0.2


def test_load_ddpg_config_accepts_normal_noise(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        ddpg_yaml(
            """
env:
  id: Pendulum-v1
  num_envs: 4
"""
        ).replace(
            "exploration: {}",
            """
  learning_starts: 100
  exploration:
    noise_type: normal
    sigma: 0.1
""",
        ),
    )

    config = load_ddpg_config(path)

    assert config.train.learning_starts == 100
    assert config.train.exploration.noise_type == "normal"
    assert config.train.exploration.sigma == 0.1


def test_ddpg_rejects_observation_normalization(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        ddpg_yaml(
            """
env:
  id: Pendulum-v1
  num_envs: 1
  observation_normalization: {}
"""
        ),
    )

    with pytest.raises(ValueError, match="not supported for DDPG"):
        load_ddpg_config(path)


def test_a3c_rejects_observation_normalization(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        a3c_yaml(
            """
env:
  id: CartPole-v1
  num_envs: 1
  observation_normalization: {}
"""
        ),
    )

    with pytest.raises(ValueError, match="not supported for A3C"):
        load_a3c_config(path)
