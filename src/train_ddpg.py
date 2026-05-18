import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config import DDPGConfig, load_ddpg_config
from ddpg import DDPG, DDPGLog
from experiment import evaluate_ddpg_policy, make_envpool_env
from metrics import JSONLMetricsLogger
from models import build_ddpg_actor_critic_model
from train_runner import (
    EnvPoolTrainingSetup,
    EnvPoolTrainSpec,
    EvaluationResult,
    parse_train_args,
    resolve_train_config,
    run_envpool_training,
)

DESCRIPTION = "Train DDPG on an EnvPool continuous-action task."


def parse_args() -> argparse.Namespace:
    return parse_train_args(DESCRIPTION)


def resolve_config(args: argparse.Namespace) -> DDPGConfig:
    config = resolve_train_config(load_ddpg_config, args)
    if not isinstance(config, DDPGConfig):
        raise TypeError("Expected DDPG config.")
    return config


def _build_setup(
    config: DDPGConfig,
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
    model = build_ddpg_actor_critic_model(
        name=config.model.name,
        observation_shape=train_env.observation_shape,
        action_spec=train_env.action_spec,
        kwargs=config.model.kwargs,
    ).to(device)
    agent = DDPG(
        train_env,
        model,
        actor_learning_rate=config.train.actor_learning_rate,
        critic_learning_rate=config.train.critic_learning_rate,
        critic_weight_decay=config.train.critic_weight_decay,
        discount_factor=config.train.discount_factor,
        soft_update_rate=config.train.soft_update_rate,
        buffer_capacity=config.train.buffer_capacity,
        batch_size=config.train.batch_size,
        learning_starts=config.train.learning_starts,
        noise_type=config.train.exploration.noise_type,
        ou_theta=config.train.exploration.theta,
        ou_sigma=config.train.exploration.sigma,
    )
    return EnvPoolTrainingSetup(
        agent=agent,
        train_env=train_env,
        eval_env=eval_env,
        plot_title=f"{config.env.id} DDPG",
        start_message=(
            f"Training {config.env.id} for at least {config.train.steps} DDPG env "
            f"steps on {device} with {config.env.num_envs} EnvPool envs. "
            f"Run directory: {run_dir}"
        ),
    )


def _train_agent(agent: DDPG, config: DDPGConfig, log_fn) -> None:
    agent.train(
        num_steps=config.train.steps,
        log_fn=log_fn,
    )


def _write_training_metrics(metrics: JSONLMetricsLogger, log: DDPGLog) -> None:
    stats = log.stats
    metrics.write(
        step=log.step,
        update=log.update,
        actor_loss=None if stats is None else stats.actor_loss,
        critic_loss=None if stats is None else stats.critic_loss,
        q_mean=None if stats is None else stats.q_mean,
        target_q_mean=None if stats is None else stats.target_q_mean,
    )


def _format_loss_line(log: DDPGLog) -> str | None:
    stats = log.stats
    if stats is None:
        return None
    return (
        f"step={log.step:6d} "
        f"actor_loss={stats.actor_loss:.4f} "
        f"critic_loss={stats.critic_loss:.4f} "
        f"q_mean={stats.q_mean:.4f} "
        f"target_q_mean={stats.target_q_mean:.4f}"
    )


def _evaluate(
    agent: DDPG,
    eval_env,
    config: DDPGConfig,
    _log: DDPGLog,
) -> EvaluationResult:
    return EvaluationResult(
        returns=evaluate_ddpg_policy(
            model=agent.online_model,
            env=eval_env,
            num_episodes=config.eval.episodes,
        )
    )


_SPEC = EnvPoolTrainSpec(
    algorithm="ddpg",
    description=DESCRIPTION,
    config_type=DDPGConfig,
    load_config=load_ddpg_config,
    build_setup=_build_setup,
    train_agent=_train_agent,
    write_training_metrics=_write_training_metrics,
    format_loss_line=_format_loss_line,
    evaluate=_evaluate,
)


def main() -> None:
    run_envpool_training(_SPEC, parse_args())


if __name__ == "__main__":
    main()
