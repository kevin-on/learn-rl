from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(ConfigModel):
    name: str = Field(min_length=1)
    run_root: str


class ObservationNormalizationConfig(ConfigModel):
    clip: PositiveFloat = 10.0
    epsilon: PositiveFloat = 1e-8


class AtariConfig(ConfigModel):
    stack_num: PositiveInt
    frame_skip: PositiveInt
    noop_max: NonNegativeInt
    episodic_life: bool
    reward_clip: bool
    img_height: PositiveInt
    img_width: PositiveInt
    gray_scale: bool


class EnvConfig(ConfigModel):
    id: str = Field(min_length=1)
    num_envs: PositiveInt
    atari: AtariConfig | None = None
    observation_normalization: ObservationNormalizationConfig | None = None


class ModelConfig(ConfigModel):
    name: str = Field(min_length=1)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class DQNExplorationConfig(ConfigModel):
    schedule: str = Field(min_length=1)
    start: NonNegativeFloat
    end: NonNegativeFloat
    decay_steps: PositiveInt


class DDPGExplorationConfig(ConfigModel):
    theta: PositiveFloat = 0.15
    sigma: NonNegativeFloat = 0.2


class DQNTrainConfig(ConfigModel):
    steps: PositiveInt
    batch_size: PositiveInt
    buffer_capacity: PositiveInt
    learning_starts: NonNegativeInt
    learning_rate: PositiveFloat
    discount_factor: Probability
    soft_update_rate: Probability
    max_grad_norm: PositiveFloat | None
    exploration: DQNExplorationConfig

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.buffer_capacity < self.batch_size:
            raise ValueError("train.buffer_capacity must be at least train.batch_size.")
        return self


class DDPGTrainConfig(ConfigModel):
    steps: PositiveInt
    batch_size: PositiveInt
    buffer_capacity: PositiveInt
    actor_learning_rate: PositiveFloat
    critic_learning_rate: PositiveFloat
    critic_weight_decay: NonNegativeFloat = 0.0
    discount_factor: Probability
    soft_update_rate: Probability
    exploration: DDPGExplorationConfig

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.buffer_capacity < self.batch_size:
            raise ValueError("train.buffer_capacity must be at least train.batch_size.")
        return self


class A2CTrainConfig(ConfigModel):
    steps: PositiveInt
    learning_rate: PositiveFloat
    value_loss_coef: NonNegativeFloat
    discount_factor: Probability
    rollout_steps: PositiveInt
    max_grad_norm: PositiveFloat | None


class A3CTrainConfig(ConfigModel):
    steps: PositiveInt
    num_workers: PositiveInt
    learning_rate: PositiveFloat
    value_loss_coef: NonNegativeFloat
    discount_factor: Probability
    rollout_steps: PositiveInt
    max_grad_norm: PositiveFloat | None
    entropy_coef: NonNegativeFloat
    rmsprop_alpha: Annotated[float, Field(ge=0.0, lt=1.0)]
    rmsprop_eps: PositiveFloat


class PPOTrainConfig(ConfigModel):
    steps: PositiveInt
    learning_rate: PositiveFloat
    discount_factor: Probability
    gae_lambda: Probability
    rollout_steps: PositiveInt
    minibatch_size: PositiveInt
    epochs: PositiveInt
    clip_coef: PositiveFloat
    value_coef: NonNegativeFloat
    entropy_coef: NonNegativeFloat
    max_grad_norm: PositiveFloat | None


class EvalConfig(ConfigModel):
    every_steps: PositiveInt
    episodes: PositiveInt
    seed: NonNegativeInt


class LoggingConfig(ConfigModel):
    loss_every_steps: PositiveInt
    save_plot: bool = True


class CheckpointConfig(ConfigModel):
    every_steps: PositiveInt | None = None


class DQNConfig(ConfigModel):
    experiment: ExperimentConfig
    env: EnvConfig
    seed: NonNegativeInt
    model: ModelConfig
    train: DQNTrainConfig
    eval: EvalConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @model_validator(mode="after")
    def validate_algorithm_support(self) -> Self:
        if self.env.observation_normalization is not None:
            raise ValueError(
                "env.observation_normalization is not supported for DQN yet; "
                "replay-buffer samples need raw-observation storage and current-stat "
                "normalization."
            )
        return self


class DDPGConfig(ConfigModel):
    experiment: ExperimentConfig
    env: EnvConfig
    seed: NonNegativeInt
    model: ModelConfig
    train: DDPGTrainConfig
    eval: EvalConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @model_validator(mode="after")
    def validate_algorithm_support(self) -> Self:
        if self.env.observation_normalization is not None:
            raise ValueError(
                "env.observation_normalization is not supported for DDPG yet; "
                "replay-buffer samples need raw-observation storage and current-stat "
                "normalization."
            )
        return self


class A2CConfig(ConfigModel):
    experiment: ExperimentConfig
    env: EnvConfig
    seed: NonNegativeInt
    model: ModelConfig
    train: A2CTrainConfig
    eval: EvalConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


class A3CConfig(ConfigModel):
    experiment: ExperimentConfig
    env: EnvConfig
    seed: NonNegativeInt
    model: ModelConfig
    train: A3CTrainConfig
    eval: EvalConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @model_validator(mode="after")
    def validate_algorithm_support(self) -> Self:
        if self.env.observation_normalization is not None:
            raise ValueError(
                "env.observation_normalization is not supported for A3C yet; "
                "A3C uses per-worker Gym envs instead of the EnvPool VecEnv wrapper."
            )
        return self


class PPOConfig(ConfigModel):
    experiment: ExperimentConfig
    env: EnvConfig
    seed: NonNegativeInt
    model: ModelConfig
    train: PPOTrainConfig
    eval: EvalConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @model_validator(mode="after")
    def validate_rollout_batch_size(self) -> Self:
        rollout_batch_size = self.env.num_envs * self.train.rollout_steps
        if self.train.minibatch_size > rollout_batch_size:
            raise ValueError(
                "train.minibatch_size must be at most "
                "env.num_envs * train.rollout_steps."
            )
        return self


type ExperimentRunConfig = DQNConfig | DDPGConfig | A2CConfig | A3CConfig | PPOConfig


def load_config(config_path: Path, overrides: list[str] | None = None) -> DQNConfig:
    return _load_run_config(config_path, DQNConfig, overrides)


def load_ddpg_config(
    config_path: Path, overrides: list[str] | None = None
) -> DDPGConfig:
    return _load_run_config(config_path, DDPGConfig, overrides)


def load_a2c_config(config_path: Path, overrides: list[str] | None = None) -> A2CConfig:
    return _load_run_config(config_path, A2CConfig, overrides)


def load_a3c_config(config_path: Path, overrides: list[str] | None = None) -> A3CConfig:
    return _load_run_config(config_path, A3CConfig, overrides)


def load_ppo_config(config_path: Path, overrides: list[str] | None = None) -> PPOConfig:
    return _load_run_config(config_path, PPOConfig, overrides)


def save_config(config: ExperimentRunConfig, config_path: Path) -> None:
    config_path.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def config_to_dict(config: ExperimentRunConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")


def _load_run_config[ConfigT: ConfigModel](
    config_path: Path,
    config_type: type[ConfigT],
    overrides: list[str] | None,
) -> ConfigT:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_config, dict):
        msg = f"Config root must be a mapping: {config_path}"
        raise ValueError(msg)

    for override in overrides or []:
        _apply_override(raw_config, override)

    return config_type.model_validate(raw_config)


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        msg = f"Override must use KEY=VALUE syntax, got: {override}"
        raise ValueError(msg)

    key, raw_value = override.split("=", maxsplit=1)
    path = key.split(".")
    if not key or any(part == "" for part in path):
        msg = f"Override key must be a dotted path, got: {key}"
        raise ValueError(msg)

    cursor = config
    for part in path[:-1]:
        next_cursor = cursor.setdefault(part, {})
        if not isinstance(next_cursor, dict):
            msg = f"Cannot set nested override through non-mapping key: {part}"
            raise ValueError(msg)
        cursor = next_cursor

    cursor[path[-1]] = yaml.safe_load(raw_value)
