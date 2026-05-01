import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from dqn import DQN, DQNLog
from task_adapter import CartPoleTaskAdapter

SEED = 123
TRAIN_STEPS = 25_000
BATCH_SIZE = 64
BUFFER_CAPACITY = 50_000
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 0.10
SOFT_UPDATE_RATE = 0.005

LOSS_LOG_EVERY_STEPS = 1_000
EVAL_EVERY_STEPS = 5_000
EVAL_EPISODES = 10


def build_q_net(num_actions: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, num_actions),
    )


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_policy(
    q_net: nn.Module,
    task_adapter: CartPoleTaskAdapter,
    num_episodes: int,
    seed: int,
) -> list[float]:
    was_training = q_net.training
    q_net.eval()
    device = next(q_net.parameters()).device
    episode_returns: list[float] = []

    for episode_index in range(num_episodes):
        observation, _info = task_adapter.env.reset(seed=seed + episode_index)
        state = task_adapter.encode_observation(observation)
        done = False
        episode_return = 0.0

        while not done:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            q_values = q_net(state_tensor)
            action_index = int(q_values.argmax(dim=1).item())
            env_action = task_adapter.action_index_to_env_action(action_index)

            observation, reward, terminated, truncated, _info = task_adapter.env.step(
                env_action
            )
            state = task_adapter.encode_observation(observation)
            episode_return += float(reward)
            done = terminated or truncated

        episode_returns.append(episode_return)

    if was_training:
        q_net.train()

    return episode_returns


def main() -> None:
    set_random_seeds(SEED)
    device = choose_device()

    train_env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    train_env.action_space.seed(SEED)
    train_adapter = CartPoleTaskAdapter(train_env)

    eval_env = gym.make("CartPole-v1")
    eval_env.action_space.seed(SEED + TRAIN_STEPS)
    eval_adapter = CartPoleTaskAdapter(eval_env)

    q_net = build_q_net(train_adapter.num_actions).to(device)
    agent = DQN(
        train_adapter,
        q_net,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        exploration_rate=EXPLORATION_RATE,
        soft_update_rate=SOFT_UPDATE_RATE,
        buffer_capacity=BUFFER_CAPACITY,
    )

    recent_returns: deque[float] = deque(maxlen=20)

    def log_training(agent: DQN, log: DQNLog) -> None:
        step = log.step_index + 1

        if log.loss is not None and step % LOSS_LOG_EVERY_STEPS == 0:
            print(f"step={step:6d} loss={log.loss:.4f}")

        if "episode" not in log.info:
            return

        episode = log.info["episode"]
        episode_return = float(episode["r"])
        episode_length = int(episode["l"])
        recent_returns.append(episode_return)
        mean_return = float(np.mean(recent_returns))
        print(
            f"step={step:6d} "
            f"train_return={episode_return:6.1f} "
            f"mean20_return={mean_return:6.1f} "
            f"episode_length={episode_length:3d}"
        )

    def run_evaluation(agent: DQN, step_index: int) -> None:
        step = step_index + 1
        if step % EVAL_EVERY_STEPS != 0:
            return

        returns = evaluate_policy(
            agent.online_q_net,
            eval_adapter,
            num_episodes=EVAL_EPISODES,
            seed=SEED + step,
        )
        print(
            f"step={step:6d} "
            f"eval_mean_return={np.mean(returns):6.1f} "
            f"eval_best_return={np.max(returns):6.1f}"
        )

    print(f"Training CartPole-v1 for {TRAIN_STEPS} steps on {device}.")
    try:
        agent.train(
            num_steps=TRAIN_STEPS,
            batch_size=BATCH_SIZE,
            env_seed=SEED,
            log_fn=log_training,
            eval_fn=run_evaluation,
        )
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
