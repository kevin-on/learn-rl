import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config import (
    A2CConfig,
    DQNConfig,
    ExperimentRunConfig,
    PPOConfig,
    config_to_dict,
)

SCHEMA_VERSION = 1
type AlgorithmName = Literal["dqn", "a2c", "ppo"]


class CheckpointPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[SCHEMA_VERSION]
    algorithm: AlgorithmName
    step: Annotated[int, Field(ge=0)]
    update: Annotated[int, Field(ge=0)] | None
    config: Mapping[str, Any]
    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    algorithm_state: Mapping[str, Any]
    observation_normalization: Mapping[str, Any] | None
    best_eval_mean_return: int | float | None
    best_step: Annotated[int, Field(ge=0)] | None


def checkpoint_paths(run_dir: Path) -> tuple[Path, Path]:
    checkpoint_dir = run_dir / "checkpoints"
    return checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"


def resume_checkpoint_path(run_dir: Path) -> Path:
    if not run_dir.is_dir():
        msg = f"--resume must point to a run directory: {run_dir}"
        raise NotADirectoryError(msg)

    checkpoint_path, _best_checkpoint_path = checkpoint_paths(run_dir)
    if not checkpoint_path.is_file():
        msg = f"Could not find resume checkpoint: {checkpoint_path}"
        raise FileNotFoundError(msg)

    return checkpoint_path


def save_checkpoint(payload: Mapping[str, Any], path: Path) -> None:
    validate_checkpoint_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(dict(payload), tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    map_location: torch.device | str | None = None,
    expected_algorithm: AlgorithmName | None = None,
) -> dict[str, Any]:
    if path.is_dir():
        msg = f"checkpoint path must be a file, got directory: {path}"
        raise IsADirectoryError(msg)
    if not path.is_file():
        msg = f"checkpoint file does not exist: {path}"
        raise FileNotFoundError(msg)

    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )
    validate_checkpoint_payload(payload, expected_algorithm=expected_algorithm)
    return payload


def build_checkpoint_payload(
    *,
    algorithm: AlgorithmName,
    config: ExperimentRunConfig,
    agent_state: Mapping[str, Any],
    observation_normalization: Mapping[str, Any] | None,
    best_eval_mean_return: float | None,
    best_step: int | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": algorithm,
        "step": int(agent_state["step"]),
        "update": agent_state["update"],
        "config": config_to_dict(config),
        "model_state": agent_state["model_state"],
        "optimizer_state": agent_state["optimizer_state"],
        "algorithm_state": agent_state["algorithm_state"],
        "observation_normalization": observation_normalization,
        "best_eval_mean_return": best_eval_mean_return,
        "best_step": best_step,
    }
    validate_checkpoint_payload(payload, expected_algorithm=algorithm)
    return payload


def validate_checkpoint_payload(
    payload: object,
    *,
    expected_algorithm: AlgorithmName | None = None,
) -> None:
    try:
        checkpoint = CheckpointPayloadModel.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid checkpoint payload: {exc}") from exc

    if expected_algorithm is not None and checkpoint.algorithm != expected_algorithm:
        msg = f"expected {expected_algorithm} checkpoint, got {checkpoint.algorithm}"
        raise ValueError(msg)


def config_from_checkpoint(payload: Mapping[str, Any]) -> ExperimentRunConfig:
    algorithm = payload["algorithm"]
    if algorithm == "dqn":
        return DQNConfig.model_validate(payload["config"])
    if algorithm == "a2c":
        return A2CConfig.model_validate(payload["config"])
    if algorithm == "ppo":
        return PPOConfig.model_validate(payload["config"])

    msg = f"unsupported checkpoint algorithm: {algorithm!r}"
    raise ValueError(msg)


def apply_resume_overrides(
    config: ExperimentRunConfig,
    *,
    overrides: list[str],
    checkpoint_step: int,
) -> ExperimentRunConfig:
    if not overrides:
        return config

    if len(overrides) != 1 or not overrides[0].startswith("train.steps="):
        raise ValueError("resume only supports --set train.steps=<larger total>.")

    raw_value = overrides[0].split("=", maxsplit=1)[1]
    value = yaml.safe_load(raw_value)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("resume train.steps override must be a positive integer.")
    if value <= checkpoint_step:
        msg = (
            "resume train.steps override must be larger than checkpoint step "
            f"({checkpoint_step})."
        )
        raise ValueError(msg)

    return config.model_copy(
        update={"train": config.train.model_copy(update={"steps": value})}
    )
