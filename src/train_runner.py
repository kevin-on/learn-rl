import argparse
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from checkpoints import (
    AlgorithmName,
    apply_resume_overrides,
    build_checkpoint_payload,
    checkpoint_paths,
    config_from_checkpoint,
    load_checkpoint,
    resume_checkpoint_path,
    save_checkpoint,
    step_checkpoint_path,
)
from config import ExperimentRunConfig, save_config
from envs import VecEnv
from experiment import choose_device, create_run_dir, set_random_seeds
from metrics import JSONLMetricsLogger
from plot_metrics import plot_metrics


@dataclass(frozen=True)
class EnvPoolTrainingSetup:
    agent: Any
    train_env: VecEnv
    eval_env: VecEnv
    plot_title: str
    start_message: str


@dataclass(frozen=True)
class EvaluationResult:
    returns: list[float]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    print_suffix: str = ""


@dataclass(frozen=True)
class EnvPoolTrainSpec:
    algorithm: AlgorithmName
    description: str
    config_type: type[ExperimentRunConfig]
    load_config: Callable[[Path, list[str] | None], ExperimentRunConfig]
    build_setup: Callable[
        [ExperimentRunConfig, torch.device, Mapping[str, Any] | None, Path],
        EnvPoolTrainingSetup,
    ]
    train_agent: Callable[[Any, ExperimentRunConfig, Callable[[Any, Any], None]], None]
    write_training_metrics: Callable[[JSONLMetricsLogger, Any], None]
    format_loss_line: Callable[[Any], str | None]
    evaluate: Callable[[Any, VecEnv, ExperimentRunConfig, Any], EvaluationResult]
    format_episode_line: Callable[[Any, Any, float], str] = (
        lambda log, episode, mean_return: default_episode_line(
            log, episode, mean_return
        )
    )
    checkpoint_observation_normalization: Callable[
        [VecEnv], Mapping[str, Any] | None
    ] = lambda _train_env: None
    after_training: Callable[[ExperimentRunConfig, Path, VecEnv], None] = (
        lambda _config, _run_dir, _train_env: None
    )


def parse_train_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a YAML experiment config.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Run directory to resume from. Loads checkpoints/last.pt.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Output directory for resolved config, metrics JSONL, and plots.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch device to use. 'auto' prefers CUDA, then CPU.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set train.steps=1000.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the metrics plot after training.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        help="Also save checkpoints/step_<step>.pt every K environment steps.",
    )
    args = parser.parse_args()
    if args.resume is None and args.config is None:
        parser.error("--config is required unless --resume is provided.")
    if args.resume is not None and args.config is not None:
        parser.error("--config cannot be used with --resume.")
    if args.resume is not None and args.run_dir is not None:
        parser.error("--run-dir cannot be used with --resume.")
    if args.checkpoint_every_steps is not None and args.checkpoint_every_steps <= 0:
        parser.error("--checkpoint-every-steps must be positive.")
    return args


def resolve_train_config(
    load_config: Callable[[Path, list[str] | None], ExperimentRunConfig],
    args: argparse.Namespace,
) -> ExperimentRunConfig:
    config = load_config(args.config, overrides=args.overrides)
    return _apply_common_cli_options(config, args)


def run_envpool_training(
    spec: EnvPoolTrainSpec,
    args: argparse.Namespace | None = None,
) -> None:
    args = parse_train_args(spec.description) if args is None else args
    device = choose_device(args.device)
    config, run_dir, checkpoint, append_metrics = _load_or_create_run(
        spec=spec,
        args=args,
        device=device,
    )
    set_random_seeds(config.seed)

    metrics_path = run_dir / "metrics.jsonl"
    last_checkpoint_path, best_checkpoint_path = checkpoint_paths(run_dir)
    setup = spec.build_setup(config, device, checkpoint, run_dir)
    agent = setup.agent
    if checkpoint is not None:
        agent.load_checkpoint_state(checkpoint)

    if config.train.steps <= agent.step:
        msg = (
            "train.steps must be larger than the checkpoint step when resuming; "
            f"got train.steps={config.train.steps}, checkpoint step={agent.step}."
        )
        raise ValueError(msg)

    recent_returns: deque[float] = deque(maxlen=20)
    next_loss_step = _next_scheduled_step(config.logging.loss_every_steps, agent.step)
    next_eval_step = _next_scheduled_step(config.eval.every_steps, agent.step)
    checkpoint_every_steps = config.checkpoint.every_steps
    next_checkpoint_step = (
        None
        if checkpoint_every_steps is None
        else _next_scheduled_step(checkpoint_every_steps, agent.step)
    )
    if checkpoint is not None and best_checkpoint_path.exists():
        best_eval_mean_return = checkpoint.get("best_eval_mean_return")
        best_step = checkpoint.get("best_step")
    else:
        best_eval_mean_return = None
        best_step = None

    def checkpoint_payload() -> dict[str, Any]:
        return build_checkpoint_payload(
            algorithm=spec.algorithm,
            config=config,
            agent_state=agent.checkpoint_state(),
            observation_normalization=spec.checkpoint_observation_normalization(
                setup.train_env
            ),
            best_eval_mean_return=best_eval_mean_return,
            best_step=best_step,
        )

    def save_last_checkpoint() -> None:
        save_checkpoint(checkpoint_payload(), last_checkpoint_path)

    def save_periodic_checkpoint(log_step: int) -> None:
        save_checkpoint(checkpoint_payload(), step_checkpoint_path(run_dir, log_step))

    def log_training(log_agent: Any, log: Any) -> None:
        nonlocal best_eval_mean_return, best_step, next_checkpoint_step
        nonlocal next_loss_step, next_eval_step

        log_step = int(log.step)
        spec.write_training_metrics(metrics, log)

        loss_line = spec.format_loss_line(log)
        if loss_line is not None and log_step >= next_loss_step:
            while next_loss_step <= log_step:
                next_loss_step += config.logging.loss_every_steps
            print(loss_line)

        for episode in _log_episodes(spec, metrics, recent_returns, log):
            print(episode)

        if log_step >= next_eval_step:
            while next_eval_step <= log_step:
                next_eval_step += config.eval.every_steps

            eval_result = spec.evaluate(log_agent, setup.eval_env, config, log)
            eval_mean_return = float(np.mean(eval_result.returns))
            eval_std_return = float(np.std(eval_result.returns))
            eval_best_return = float(np.max(eval_result.returns))
            metrics.write(
                step=log_step,
                **dict(eval_result.metrics),
                eval_seed=config.eval.seed,
                eval_mean_return=eval_mean_return,
                eval_std_return=eval_std_return,
                eval_best_return=eval_best_return,
            )
            print(
                f"step={log_step:6d} "
                f"eval_mean_return={eval_mean_return:6.1f} "
                f"eval_std_return={eval_std_return:6.1f} "
                f"eval_best_return={eval_best_return:6.1f}"
                f"{eval_result.print_suffix}"
            )
            if (
                best_eval_mean_return is None
                or eval_mean_return > best_eval_mean_return
            ):
                best_eval_mean_return = eval_mean_return
                best_step = log_step
                save_checkpoint(checkpoint_payload(), best_checkpoint_path)
            save_last_checkpoint()

        if next_checkpoint_step is not None and log_step >= next_checkpoint_step:
            save_periodic_checkpoint(log_step)
            while next_checkpoint_step <= log_step:
                next_checkpoint_step += checkpoint_every_steps

    print(setup.start_message)
    with JSONLMetricsLogger(metrics_path, append=append_metrics) as metrics:
        try:
            spec.train_agent(agent, config, log_training)
        finally:
            try:
                save_last_checkpoint()
            finally:
                setup.train_env.close()
                setup.eval_env.close()

    spec.after_training(config, run_dir, setup.train_env)

    if config.logging.save_plot:
        plot_path = run_dir / "metrics.png"
        plot_metrics(metrics_path, plot_path, title=setup.plot_title)
        print(f"Saved metrics plot to {plot_path}")


def default_episode_line(log: Any, episode: Any, mean_return: float) -> str:
    return (
        f"step={log.step:6d} "
        f"env={episode.env_id:3d} "
        f"train_return={episode.episode_return:6.1f} "
        f"mean20_return={mean_return:6.1f} "
        f"episode_length={episode.episode_length:4d}"
    )


def _load_or_create_run(
    *,
    spec: EnvPoolTrainSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ExperimentRunConfig, Path, Mapping[str, Any] | None, bool]:
    if args.resume is not None:
        run_dir = args.resume
        checkpoint_path = resume_checkpoint_path(run_dir)
        checkpoint = load_checkpoint(
            checkpoint_path,
            map_location=device,
            expected_algorithm=spec.algorithm,
        )
        config = config_from_checkpoint(checkpoint)
        if not isinstance(config, spec.config_type):
            msg = f"Expected {spec.config_type.__name__} config in checkpoint."
            raise TypeError(msg)
        config = apply_resume_overrides(
            config,
            overrides=args.overrides,
            checkpoint_step=int(checkpoint["step"]),
        )
        config = _apply_common_cli_options(config, args)
        return config, run_dir, checkpoint, True

    config = spec.load_config(args.config, overrides=args.overrides)
    config = _apply_common_cli_options(config, args)
    run_dir = create_run_dir(config, args.run_dir)
    save_config(config, run_dir / "config.yaml")
    return config, run_dir, None, False


def _apply_common_cli_options(
    config: ExperimentRunConfig,
    args: argparse.Namespace,
) -> ExperimentRunConfig:
    if args.no_plot:
        config = config.model_copy(
            update={"logging": config.logging.model_copy(update={"save_plot": False})}
        )
    if args.checkpoint_every_steps is not None:
        config = config.model_copy(
            update={
                "checkpoint": config.checkpoint.model_copy(
                    update={"every_steps": args.checkpoint_every_steps}
                )
            }
        )
    return config


def _next_scheduled_step(interval: int, current_step: int) -> int:
    if interval <= 0:
        raise ValueError("interval must be positive.")
    if current_step < 0:
        raise ValueError("current_step must be non-negative.")
    return ((current_step // interval) + 1) * interval


def _log_episodes(
    spec: EnvPoolTrainSpec,
    metrics: JSONLMetricsLogger,
    recent_returns: deque[float],
    log: Any,
) -> Sequence[str]:
    lines: list[str] = []
    for episode in log.episodes:
        recent_returns.append(episode.episode_return)
        mean_return = float(np.mean(recent_returns))
        metrics.write(
            step=log.step,
            env_id=episode.env_id,
            train_episode_return=episode.episode_return,
            train_episode_return_mean20=mean_return,
            train_episode_length=episode.episode_length,
        )
        lines.append(spec.format_episode_line(log, episode, mean_return))
    return lines
