import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config import DQNConfig, load_config
from dqn import DQN, DQNLog
from experiment import evaluate_q_policy, make_envpool_env
from metrics import JSONLMetricsLogger
from models import build_q_model
from schedules import ExplorationRateSchedule
from train_runner import (
    EnvPoolTrainingSetup,
    EnvPoolTrainSpec,
    EvaluationResult,
    parse_train_args,
    resolve_train_config,
    run_envpool_training,
)

DESCRIPTION = "Train DQN on an EnvPool discrete-action task."


def parse_args() -> argparse.Namespace:
    return parse_train_args(DESCRIPTION)


def resolve_config(args: argparse.Namespace) -> DQNConfig:
    config = resolve_train_config(load_config, args)
    if not isinstance(config, DQNConfig):
        raise TypeError("Expected DQN config.")
    return config


def _exploration_schedule(config: DQNConfig) -> ExplorationRateSchedule:
    return ExplorationRateSchedule(
        schedule=config.train.exploration.schedule,
        start=config.train.exploration.start,
        end=config.train.exploration.end,
        decay_steps=config.train.exploration.decay_steps,
    )


def _build_setup(
    config: DQNConfig,
    device: Any,
    _checkpoint: Mapping[str, Any] | None,
    run_dir: Path,
) -> EnvPoolTrainingSetup:
    train_env = make_envpool_env(
        config,
        num_envs=config.env.num_envs,
        seed=config.seed,
    )
    eval_env = make_envpool_env(
        config,
        num_envs=1,
        seed=config.eval.seed,
        evaluation=True,
    )
    q_net = build_q_model(
        name=config.model.name,
        observation_shape=train_env.observation_shape,
        num_actions=train_env.num_actions,
        kwargs=config.model.kwargs,
    ).to(device)
    agent = DQN(
        train_env,
        q_net,
        learning_rate=config.train.learning_rate,
        discount_factor=config.train.discount_factor,
        soft_update_rate=config.train.soft_update_rate,
        buffer_capacity=config.train.buffer_capacity,
        batch_size=config.train.batch_size,
        learning_starts=config.train.learning_starts,
        max_grad_norm=config.train.max_grad_norm,
    )
    return EnvPoolTrainingSetup(
        agent=agent,
        train_env=train_env,
        eval_env=eval_env,
        plot_title=f"{config.env.id} DQN",
        start_message=(
            f"Training {config.env.id} for at least {config.train.steps} DQN env "
            f"steps on {device} with {config.env.num_envs} EnvPool envs. "
            f"Run directory: {run_dir}"
        ),
    )


def _train_agent(
    agent: DQN,
    config: DQNConfig,
    log_fn,
) -> None:
    exploration_schedule = _exploration_schedule(config)
    agent.train(
        num_steps=config.train.steps,
        exploration_rate_fn=exploration_schedule.value,
        log_fn=log_fn,
    )


def _write_training_metrics(metrics: JSONLMetricsLogger, log: DQNLog) -> None:
    metrics.write(
        step=log.step,
        loss=log.loss,
        grad_norm=log.grad_norm,
        epsilon=log.exploration_rate,
    )


def _format_loss_line(log: DQNLog) -> str | None:
    if log.loss is None:
        return None
    grad_norm_text = "" if log.grad_norm is None else f" grad_norm={log.grad_norm:.4f}"
    return (
        f"step={log.step:6d} loss={log.loss:.4f}{grad_norm_text} "
        f"epsilon={log.exploration_rate:.3f}"
    )


def _format_episode_line(
    log: DQNLog,
    episode: Any,
    mean_return: float,
) -> str:
    return (
        f"step={log.step:6d} "
        f"env={episode.env_id:3d} "
        f"train_return={episode.episode_return:6.1f} "
        f"mean20_return={mean_return:6.1f} "
        f"episode_length={episode.episode_length:4d} "
        f"epsilon={log.exploration_rate:.3f}"
    )


def _evaluate(
    agent: DQN,
    eval_env,
    config: DQNConfig,
    log: DQNLog,
) -> EvaluationResult:
    returns = evaluate_q_policy(
        q_net=agent.online_q_net,
        env=eval_env,
        num_episodes=config.eval.episodes,
    )
    return EvaluationResult(
        returns=returns,
        metrics={"epsilon": log.exploration_rate},
        print_suffix=f" epsilon={log.exploration_rate:.3f}",
    )


_SPEC = EnvPoolTrainSpec(
    algorithm="dqn",
    description=DESCRIPTION,
    config_type=DQNConfig,
    load_config=load_config,
    build_setup=_build_setup,
    train_agent=_train_agent,
    write_training_metrics=_write_training_metrics,
    format_loss_line=_format_loss_line,
    evaluate=_evaluate,
    format_episode_line=_format_episode_line,
)


def main() -> None:
    run_envpool_training(_SPEC, parse_args())


if __name__ == "__main__":
    main()
