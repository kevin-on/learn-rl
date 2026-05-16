from a2c import A2C
from ddpg import DDPG
from dqn import DQN
from envs import EnvPoolVecEnv
from models import (
    build_actor_critic_model,
    build_ddpg_actor_critic_model,
    build_q_model,
)
from ppo import PPO


def test_dqn_cartpole_tiny_smoke() -> None:
    env = EnvPoolVecEnv(env_id="CartPole-v1", num_envs=1, seed=1)
    try:
        model = build_q_model(
            name="mlp",
            observation_shape=env.observation_shape,
            num_actions=env.num_actions,
            kwargs={"hidden_sizes": [8]},
        )
        agent = DQN(
            env,
            model,
            learning_rate=1e-3,
            discount_factor=0.99,
            soft_update_rate=0.005,
            buffer_capacity=16,
            batch_size=4,
            learning_starts=0,
            max_grad_norm=1.0,
        )
        agent.train(num_steps=4, exploration_rate_fn=lambda _step: 1.0)
    finally:
        env.close()


def test_a2c_cartpole_tiny_smoke() -> None:
    env = EnvPoolVecEnv(env_id="CartPole-v1", num_envs=2, seed=2)
    try:
        model = build_actor_critic_model(
            name="discrete_mlp",
            observation_shape=env.observation_shape,
            action_spec=env.action_spec,
            kwargs={"hidden_sizes": [8]},
        )
        agent = A2C(
            env,
            model,
            learning_rate=1e-3,
            value_loss_coef=0.5,
            discount_factor=0.99,
            rollout_steps=2,
            max_grad_norm=1.0,
        )
        agent.train(num_steps=4)
    finally:
        env.close()


def test_ppo_cartpole_tiny_smoke() -> None:
    env = EnvPoolVecEnv(env_id="CartPole-v1", num_envs=2, seed=3)
    try:
        model = build_actor_critic_model(
            name="discrete_mlp",
            observation_shape=env.observation_shape,
            action_spec=env.action_spec,
            kwargs={"hidden_sizes": [8]},
        )
        agent = PPO(
            env,
            model,
            learning_rate=1e-3,
            rollout_steps=2,
            minibatch_size=4,
            epochs=1,
            discount_factor=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=1.0,
        )
        agent.train(num_steps=4)
    finally:
        env.close()


def test_a2c_pendulum_tiny_smoke() -> None:
    env = EnvPoolVecEnv(env_id="Pendulum-v1", num_envs=2, seed=4)
    try:
        model = build_actor_critic_model(
            name="continuous_mlp",
            observation_shape=env.observation_shape,
            action_spec=env.action_spec,
            kwargs={
                "hidden_sizes": [8],
                "init_log_std": 0.0,
                "log_std_min": -20.0,
                "log_std_max": 2.0,
            },
        )
        agent = A2C(
            env,
            model,
            learning_rate=1e-3,
            value_loss_coef=0.5,
            discount_factor=0.99,
            rollout_steps=2,
            max_grad_norm=1.0,
        )
        agent.train(num_steps=4)
    finally:
        env.close()


def test_ddpg_pendulum_tiny_smoke() -> None:
    env = EnvPoolVecEnv(env_id="Pendulum-v1", num_envs=2, seed=6)
    try:
        model = build_ddpg_actor_critic_model(
            name="ddpg_mlp",
            observation_shape=env.observation_shape,
            action_spec=env.action_spec,
            kwargs={"hidden_sizes": [8, 8]},
        )
        agent = DDPG(
            env,
            model,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            critic_weight_decay=0.0,
            discount_factor=0.99,
            soft_update_rate=0.005,
            buffer_capacity=16,
            batch_size=4,
        )
        agent.train(num_steps=4)
    finally:
        env.close()


def test_ppo_pendulum_tiny_smoke() -> None:
    env = EnvPoolVecEnv(env_id="Pendulum-v1", num_envs=2, seed=5)
    try:
        model = build_actor_critic_model(
            name="continuous_mlp",
            observation_shape=env.observation_shape,
            action_spec=env.action_spec,
            kwargs={
                "hidden_sizes": [8],
                "init_log_std": 0.0,
                "log_std_min": -20.0,
                "log_std_max": 2.0,
            },
        )
        agent = PPO(
            env,
            model,
            learning_rate=1e-3,
            rollout_steps=2,
            minibatch_size=4,
            epochs=1,
            discount_factor=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            max_grad_norm=1.0,
        )
        agent.train(num_steps=4)
    finally:
        env.close()
