import argparse
from pathlib import Path

import numpy as np

from checkpoints import config_from_checkpoint, load_checkpoint
from config import A2CConfig, DQNConfig, PPOConfig
from experiment import (
    choose_device,
    evaluate_actor_critic_policy,
    evaluate_q_policy,
    make_envpool_env,
    running_mean_std_from_state,
    set_random_seeds,
)
from models import build_actor_critic_model, build_q_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DQN, A2C, or PPO checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to a .pt checkpoint file.")
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of evaluation episodes. Defaults to the checkpoint config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Evaluation seed. Defaults to the checkpoint config.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch device to use. 'auto' prefers CUDA, then CPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be non-negative.")

    device = choose_device(args.device)
    checkpoint_path = args.checkpoint
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = config_from_checkpoint(checkpoint)
    episodes = args.episodes or config.eval.episodes
    seed = args.seed if args.seed is not None else config.eval.seed
    set_random_seeds(seed)

    if isinstance(config, DQNConfig):
        returns = evaluate_dqn_checkpoint(
            checkpoint=checkpoint,
            config=config,
            episodes=episodes,
            seed=seed,
            device=device,
        )
    elif isinstance(config, A2CConfig | PPOConfig):
        returns = evaluate_actor_critic_checkpoint(
            checkpoint=checkpoint,
            config=config,
            episodes=episodes,
            seed=seed,
            device=device,
        )
    else:
        msg = f"Unsupported checkpoint config type: {type(config).__name__}"
        raise TypeError(msg)

    eval_mean_return = float(np.mean(returns))
    eval_std_return = float(np.std(returns))
    eval_best_return = float(np.max(returns))
    print(f"checkpoint={checkpoint_path}")
    print(f"algorithm={checkpoint['algorithm']} step={checkpoint['step']}")
    print(f"episodes={episodes} seed={seed}")
    print(
        f"eval_mean_return={eval_mean_return:.3f} "
        f"eval_std_return={eval_std_return:.3f} "
        f"eval_best_return={eval_best_return:.3f}"
    )


def evaluate_dqn_checkpoint(
    *,
    checkpoint: dict,
    config: DQNConfig,
    episodes: int,
    seed: int,
    device,
) -> list[float]:
    env = make_envpool_env(
        config,
        num_envs=1,
        seed=seed,
        evaluation=True,
    )
    try:
        q_net = build_q_model(
            name=config.model.name,
            observation_shape=env.observation_shape,
            num_actions=env.num_actions,
            kwargs=config.model.kwargs,
        ).to(device)
        q_net.load_state_dict(checkpoint["model_state"])
        return evaluate_q_policy(
            q_net=q_net,
            env=env,
            num_episodes=episodes,
        )
    finally:
        env.close()


def evaluate_actor_critic_checkpoint(
    *,
    checkpoint: dict,
    config: A2CConfig | PPOConfig,
    episodes: int,
    seed: int,
    device,
) -> list[float]:
    observation_rms = (
        None
        if checkpoint["observation_normalization"] is None
        else running_mean_std_from_state(checkpoint["observation_normalization"])
    )
    env = make_envpool_env(
        config,
        num_envs=1,
        seed=seed,
        evaluation=True,
        observation_rms=observation_rms,
    )
    try:
        model = build_actor_critic_model(
            name=config.model.name,
            observation_shape=env.observation_shape,
            action_spec=env.action_spec,
            kwargs=config.model.kwargs,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        return evaluate_actor_critic_policy(
            model=model,
            env=env,
            num_episodes=episodes,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
