import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config import PPOConfig, load_ppo_config
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
from ppo import PPO, PPOLog
from train_runner import (
    EnvPoolTrainingSetup,
    EnvPoolTrainSpec,
    EvaluationResult,
    parse_train_args,
    resolve_train_config,
    run_envpool_training,
)

DESCRIPTION = "Train clipped PPO with GAE on EnvPool Gymnasium environments."


def parse_args() -> argparse.Namespace:
    return parse_train_args(DESCRIPTION)


def resolve_config(args: argparse.Namespace) -> PPOConfig:
    config = resolve_train_config(load_ppo_config, args)
    if not isinstance(config, PPOConfig):
        raise TypeError("Expected PPO config.")
    return config


def _build_setup(
    config: PPOConfig,
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
    agent = PPO(
        train_env,
        model,
        learning_rate=config.train.learning_rate,
        rollout_steps=config.train.rollout_steps,
        minibatch_size=config.train.minibatch_size,
        epochs=config.train.epochs,
        discount_factor=config.train.discount_factor,
        gae_lambda=config.train.gae_lambda,
        clip_coef=config.train.clip_coef,
        value_coef=config.train.value_coef,
        entropy_coef=config.train.entropy_coef,
        max_grad_norm=config.train.max_grad_norm,
        normalize_advantages=config.train.normalize_advantages,
    )
    return EnvPoolTrainingSetup(
        agent=agent,
        train_env=train_env,
        eval_env=eval_env,
        plot_title=f"{config.env.id} PPO",
        start_message=(
            f"Training {config.env.id} for at least {config.train.steps} environment "
            f"steps on {device} with {config.env.num_envs} EnvPool envs. "
            f"Run directory: {run_dir}"
        ),
    )


def _train_agent(agent: PPO, config: PPOConfig, log_fn) -> None:
    agent.train(num_steps=config.train.steps, log_fn=log_fn)


def _write_training_metrics(metrics: JSONLMetricsLogger, log: PPOLog) -> None:
    metrics.write(
        step=log.step,
        update=log.update,
        loss=log.stats.loss,
        policy_loss=log.stats.policy_loss,
        value_loss=log.stats.value_loss,
        entropy=log.stats.entropy,
        clip_fraction=log.stats.clip_fraction,
        grad_norm=log.stats.grad_norm,
        rollout_steps=log.rollout_steps,
        approx_kl=log.stats.approx_kl,
    )


def _format_loss_line(log: PPOLog) -> str | None:
    grad_norm_text = (
        "" if log.stats.grad_norm is None else f" grad_norm={log.stats.grad_norm:.4f}"
    )
    return (
        f"step={log.step:6d} "
        f"loss={log.stats.loss:.4f} "
        f"policy_loss={log.stats.policy_loss:.4f} "
        f"value_loss={log.stats.value_loss:.4f} "
        f"entropy={log.stats.entropy:.4f} "
        f"clip_fraction={log.stats.clip_fraction:.3f}"
        f" approx_kl={log.stats.approx_kl:.4f}"
        f"{grad_norm_text}"
    )


def _evaluate(
    agent: PPO,
    eval_env,
    config: PPOConfig,
    _log: PPOLog,
) -> EvaluationResult:
    return EvaluationResult(
        returns=evaluate_actor_critic_policy(
            model=agent.model,
            env=eval_env,
            num_episodes=config.eval.episodes,
        )
    )


def _after_training(_config: PPOConfig, run_dir: Path, train_env: VecEnv) -> None:
    obs_norm_path = save_observation_normalization_stats(
        train_env, run_dir / "observation_normalization.npz"
    )
    if obs_norm_path is not None:
        print(f"Saved observation normalization stats to {obs_norm_path}")


_SPEC = EnvPoolTrainSpec(
    algorithm="ppo",
    description=DESCRIPTION,
    config_type=PPOConfig,
    load_config=load_ppo_config,
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
