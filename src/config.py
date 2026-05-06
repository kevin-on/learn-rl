from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    run_root: str = "runs"


@dataclass(frozen=True)
class EnvConfig:
    id: str


@dataclass(frozen=True)
class ModelConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])


@dataclass(frozen=True)
class DQNTrainConfig:
    steps: int = 25_000
    batch_size: int = 64
    buffer_capacity: int = 50_000
    learning_starts: int = 0
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    soft_update_rate: float = 0.005
    max_grad_norm: float | None = 10.0


@dataclass(frozen=True)
class A2CTrainConfig:
    steps: int = 25_000
    policy_learning_rate: float = 1e-3
    value_learning_rate: float = 1e-3
    discount_factor: float = 0.99
    rollout_steps: int = 5
    max_grad_norm: float | None = 10.0


@dataclass(frozen=True)
class ExplorationConfig:
    schedule: str = "linear"
    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 15_000


@dataclass(frozen=True)
class EvalConfig:
    every_steps: int = 5_000
    episodes: int = 10
    seed: int = 10_000


@dataclass(frozen=True)
class LoggingConfig:
    loss_every_steps: int = 1_000
    save_plot: bool = True


@dataclass(frozen=True)
class DQNConfig:
    experiment: ExperimentConfig
    env: EnvConfig
    seed: int = 123
    model: ModelConfig = field(default_factory=ModelConfig)
    train: DQNTrainConfig = field(default_factory=DQNTrainConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass(frozen=True)
class A2CConfig:
    experiment: ExperimentConfig
    env: EnvConfig
    seed: int = 123
    model: ModelConfig = field(default_factory=ModelConfig)
    train: A2CTrainConfig = field(default_factory=A2CTrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


type ExperimentRunConfig = DQNConfig | A2CConfig


def load_config(config_path: Path, overrides: list[str] | None = None) -> DQNConfig:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_config, dict):
        msg = f"Config root must be a mapping: {config_path}"
        raise ValueError(msg)

    for override in overrides or []:
        _apply_override(raw_config, override)

    config = _from_dict(DQNConfig, raw_config, path="config")
    _validate_dqn_config(config)
    return config


def load_a2c_config(config_path: Path, overrides: list[str] | None = None) -> A2CConfig:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_config, dict):
        msg = f"Config root must be a mapping: {config_path}"
        raise ValueError(msg)

    for override in overrides or []:
        _apply_override(raw_config, override)

    config = _from_dict(A2CConfig, raw_config, path="config")
    _validate_a2c_config(config)
    return config


def save_config(config: ExperimentRunConfig, config_path: Path) -> None:
    config_path.write_text(
        yaml.safe_dump(asdict(config), sort_keys=False),
        encoding="utf-8",
    )


def config_to_dict(config: ExperimentRunConfig) -> dict[str, Any]:
    return asdict(config)


def _from_dict(config_type: type[Any], data: Any, path: str) -> Any:
    if not is_dataclass(config_type):
        return data

    if not isinstance(data, dict):
        msg = f"{path} must be a mapping."
        raise ValueError(msg)

    field_by_name = {field.name: field for field in fields(config_type)}
    unknown_keys = sorted(set(data) - set(field_by_name))
    if unknown_keys:
        keys = ", ".join(unknown_keys)
        msg = f"Unknown config key(s) at {path}: {keys}"
        raise ValueError(msg)

    required_keys = {
        name
        for name, field_info in field_by_name.items()
        if field_info.default is MISSING and field_info.default_factory is MISSING
    }
    missing_keys = sorted(required_keys - set(data))
    if missing_keys:
        keys = ", ".join(missing_keys)
        msg = f"Missing config key(s) at {path}: {keys}"
        raise ValueError(msg)

    kwargs: dict[str, Any] = {}
    for name, field_info in field_by_name.items():
        if name not in data:
            continue

        value = data[name]
        if is_dataclass(field_info.type):
            value = _from_dict(field_info.type, value, path=f"{path}.{name}")
        kwargs[name] = value

    return config_type(**kwargs)


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


def _validate_common_config(config: ExperimentRunConfig) -> None:
    if config.seed < 0:
        raise ValueError("seed must be non-negative.")

    if not config.model.hidden_sizes:
        raise ValueError("model.hidden_sizes must not be empty.")
    if any(hidden_size <= 0 for hidden_size in config.model.hidden_sizes):
        raise ValueError("model.hidden_sizes values must be positive.")

    if config.eval.every_steps <= 0:
        raise ValueError("eval.every_steps must be positive.")
    if config.eval.episodes <= 0:
        raise ValueError("eval.episodes must be positive.")
    if config.eval.seed < 0:
        raise ValueError("eval.seed must be non-negative.")

    if config.logging.loss_every_steps <= 0:
        raise ValueError("logging.loss_every_steps must be positive.")


def _validate_dqn_config(config: DQNConfig) -> None:
    _validate_common_config(config)

    if config.train.steps <= 0:
        raise ValueError("train.steps must be positive.")
    if config.train.batch_size <= 0:
        raise ValueError("train.batch_size must be positive.")
    if config.train.buffer_capacity < config.train.batch_size:
        raise ValueError("train.buffer_capacity must be at least train.batch_size.")
    if config.train.learning_starts < 0:
        raise ValueError("train.learning_starts must be non-negative.")
    if not 0.0 <= config.train.discount_factor <= 1.0:
        raise ValueError("train.discount_factor must be in [0, 1].")
    if config.train.learning_rate <= 0.0:
        raise ValueError("train.learning_rate must be positive.")
    if not 0.0 <= config.train.soft_update_rate <= 1.0:
        raise ValueError("train.soft_update_rate must be in [0, 1].")
    if config.train.max_grad_norm is not None and config.train.max_grad_norm <= 0.0:
        raise ValueError("train.max_grad_norm must be positive or null.")


def _validate_a2c_config(config: A2CConfig) -> None:
    _validate_common_config(config)

    if config.train.steps <= 0:
        raise ValueError("train.steps must be positive.")
    if config.train.policy_learning_rate <= 0.0:
        raise ValueError("train.policy_learning_rate must be positive.")
    if config.train.value_learning_rate <= 0.0:
        raise ValueError("train.value_learning_rate must be positive.")
    if not 0.0 <= config.train.discount_factor <= 1.0:
        raise ValueError("train.discount_factor must be in [0, 1].")
    if config.train.rollout_steps <= 0:
        raise ValueError("train.rollout_steps must be positive.")
    if config.train.max_grad_norm is not None and config.train.max_grad_norm <= 0.0:
        raise ValueError("train.max_grad_norm must be positive or null.")
