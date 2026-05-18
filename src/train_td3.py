import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config import TD3Config, load_td3_config
from experiment import evaluate_ddpg_policy, make_envpool_env
from metrics import JSONLMetricsLogger
from models import build_td3_actor_critic_model
from td3 import TD3, TD3Log
from train_runner import (
    EnvPoolTrainingSetup,
    EnvPoolTrainSpec,
    EvaluationResult,
    parse_train_args,
    resolve_train_config,
    run_envpool_training,
)

DESCRIPTION = "Train TD3 on an EnvPool continuous-action task."


def parse_args() -> argparse.Namespace:
    return parse_train_args(DESCRIPTION)


def resolve_config(args: argparse.Namespace) -> TD3Config:
    config = resolve_train_config(load_td3_config, args)
    if not isinstance(config, TD3Config):
        raise TypeError("Expected TD3 config.")
    return config


def _build_setup(
    config: TD3Config,
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
    model = build_td3_actor_critic_model(
        name=config.model.name,
        observation_shape=train_env.observation_shape,
        action_spec=train_env.action_spec,
        kwargs=config.model.kwargs,
    ).to(device)
    agent = TD3(
        train_env,
        model,
        actor_learning_rate=config.train.actor_learning_rate,
        critic_learning_rate=config.train.critic_learning_rate,
        discount_factor=config.train.discount_factor,
        soft_update_rate=config.train.soft_update_rate,
        buffer_capacity=config.train.buffer_capacity,
        batch_size=config.train.batch_size,
        learning_starts=config.train.learning_starts,
        exploration_sigma=config.train.exploration.sigma,
        target_policy_noise=config.train.target_policy_noise,
        target_noise_clip=config.train.target_noise_clip,
        policy_delay=config.train.policy_delay,
    )
    return EnvPoolTrainingSetup(
        agent=agent,
        train_env=train_env,
        eval_env=eval_env,
        plot_title=f"{config.env.id} TD3",
        start_message=(
            f"Training {config.env.id} for at least {config.train.steps} TD3 env "
            f"steps on {device} with {config.env.num_envs} EnvPool envs. "
            f"Run directory: {run_dir}"
        ),
    )


def _train_agent(agent: TD3, config: TD3Config, log_fn) -> None:
    agent.train(
        num_steps=config.train.steps,
        log_fn=log_fn,
    )


def _write_training_metrics(metrics: JSONLMetricsLogger, log: TD3Log) -> None:
    stats = log.stats
    metrics.write(
        step=log.step,
        update=log.update,
        actor_loss=None if stats is None else stats.actor_loss,
        critic_loss=None if stats is None else stats.critic_loss,
        critic1_loss=None if stats is None else stats.critic1_loss,
        critic2_loss=None if stats is None else stats.critic2_loss,
        q1_mean=None if stats is None else stats.q1_mean,
        q2_mean=None if stats is None else stats.q2_mean,
        target_q_mean=None if stats is None else stats.target_q_mean,
    )


def _format_loss_line(log: TD3Log) -> str | None:
    stats = log.stats
    if stats is None:
        return None
    actor_loss = "n/a" if stats.actor_loss is None else f"{stats.actor_loss:.4f}"
    return (
        f"step={log.step:6d} "
        f"actor_loss={actor_loss} "
        f"critic_loss={stats.critic_loss:.4f} "
        f"q1_mean={stats.q1_mean:.4f} "
        f"q2_mean={stats.q2_mean:.4f} "
        f"target_q_mean={stats.target_q_mean:.4f}"
    )


def _evaluate(
    agent: TD3,
    eval_env,
    config: TD3Config,
    _log: TD3Log,
) -> EvaluationResult:
    return EvaluationResult(
        returns=evaluate_ddpg_policy(
            model=agent.online_model,
            env=eval_env,
            num_episodes=config.eval.episodes,
        )
    )


_SPEC = EnvPoolTrainSpec(
    algorithm="td3",
    description=DESCRIPTION,
    config_type=TD3Config,
    load_config=load_td3_config,
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
