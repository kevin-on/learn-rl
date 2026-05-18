import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from a2c import A2C, A2CLog
from config import A2CConfig, load_a2c_config
from envs import VecEnv
from experiment import (
    evaluate_actor_critic_policy,
    make_envpool_env,
    observation_normalization_state,
    observation_normalization_stats,
    running_mean_std_from_state,
    save_observation_normalization_stats,
)
from metrics import JSONLMetricsLogger
from models import build_actor_critic_model
from train_runner import (
    EnvPoolTrainingSetup,
    EnvPoolTrainSpec,
    EvaluationResult,
    parse_train_args,
    resolve_train_config,
    run_envpool_training,
)

DESCRIPTION = "Train A2C on an EnvPool vector task."


def parse_args() -> argparse.Namespace:
    return parse_train_args(DESCRIPTION)


def resolve_config(args: argparse.Namespace) -> A2CConfig:
    config = resolve_train_config(load_a2c_config, args)
    if not isinstance(config, A2CConfig):
        raise TypeError("Expected A2C config.")
    return config


def _build_setup(
    config: A2CConfig,
    device: Any,
    checkpoint: Mapping[str, Any] | None,
    run_dir: Path,
) -> EnvPoolTrainingSetup:
    observation_rms = (
        None
        if checkpoint is None or checkpoint["observation_normalization"] is None
        else running_mean_std_from_state(checkpoint["observation_normalization"])
    )
    train_env = make_envpool_env(
        config,
        num_envs=config.env.num_envs,
        seed=config.seed,
        observation_rms=observation_rms,
    )
    model = build_actor_critic_model(
        name=config.model.name,
        observation_shape=train_env.observation_shape,
        action_spec=train_env.action_spec,
        kwargs=config.model.kwargs,
    ).to(device)
    eval_env = make_envpool_env(
        config,
        num_envs=1,
        seed=config.eval.seed,
        evaluation=True,
        observation_rms=observation_normalization_stats(train_env),
    )
    agent = A2C(
        train_env,
        model,
        learning_rate=config.train.learning_rate,
        value_loss_coef=config.train.value_loss_coef,
        discount_factor=config.train.discount_factor,
        rollout_steps=config.train.rollout_steps,
        max_grad_norm=config.train.max_grad_norm,
    )
    return EnvPoolTrainingSetup(
        agent=agent,
        train_env=train_env,
        eval_env=eval_env,
        plot_title=f"{config.env.id} A2C",
        start_message=(
            f"Training {config.env.id} for at least {config.train.steps} A2C env "
            f"steps on {device} with {config.env.num_envs} EnvPool envs. "
            f"Run directory: {run_dir}"
        ),
    )


def _train_agent(agent: A2C, config: A2CConfig, log_fn) -> None:
    agent.train(num_steps=config.train.steps, log_fn=log_fn)


def _write_training_metrics(metrics: JSONLMetricsLogger, log: A2CLog) -> None:
    metrics.write(
        step=log.step,
        update=log.update,
        loss=log.loss,
        policy_loss=log.policy_loss,
        value_loss=log.value_loss,
        entropy=log.entropy,
        grad_norm=log.grad_norm,
        rollout_steps=log.rollout_steps,
    )


def _format_loss_line(log: A2CLog) -> str | None:
    grad_norm_text = "" if log.grad_norm is None else f" grad_norm={log.grad_norm:.4f}"
    return (
        f"step={log.step:6d} "
        f"loss={log.loss:.4f} "
        f"policy_loss={log.policy_loss:.4f} "
        f"value_loss={log.value_loss:.4f} "
        f"entropy={log.entropy:.4f}"
        f"{grad_norm_text}"
    )


def _evaluate(
    agent: A2C,
    eval_env,
    config: A2CConfig,
    _log: A2CLog,
) -> EvaluationResult:
    return EvaluationResult(
        returns=evaluate_actor_critic_policy(
            model=agent.model,
            env=eval_env,
            num_episodes=config.eval.episodes,
        )
    )


def _after_training(_config: A2CConfig, run_dir: Path, train_env: VecEnv) -> None:
    obs_norm_path = save_observation_normalization_stats(
        train_env, run_dir / "observation_normalization.npz"
    )
    if obs_norm_path is not None:
        print(f"Saved observation normalization stats to {obs_norm_path}")


_SPEC = EnvPoolTrainSpec(
    algorithm="a2c",
    description=DESCRIPTION,
    config_type=A2CConfig,
    load_config=load_a2c_config,
    build_setup=_build_setup,
    train_agent=_train_agent,
    write_training_metrics=_write_training_metrics,
    format_loss_line=_format_loss_line,
    evaluate=_evaluate,
    checkpoint_observation_normalization=observation_normalization_state,
    after_training=_after_training,
)


def main() -> None:
    run_envpool_training(_SPEC, parse_args())


if __name__ == "__main__":
    main()
