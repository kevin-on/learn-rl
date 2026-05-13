# learn-rl

Small reinforcement-learning experiments for Gymnasium-compatible EnvPool tasks.

Implemented algorithms:

- DQN
- A2C
- A3C
- PPO

## Train

Each algorithm has its own entrypoint:

```sh
uv run python src/train_dqn.py --config configs/cartpole_dqn.yaml
uv run python src/train_a2c.py --config configs/cartpole_a2c.yaml
uv run python src/train_a3c.py --config configs/cartpole_a3c.yaml
uv run python src/train_ppo.py --config configs/cartpole_ppo.yaml
```

DQN, A2C, and PPO train through a shared EnvPool vector-environment interface.
A3C keeps its worker-process training loop, while sharing the common model and
math utilities where practical.

Each run writes:

- `config.yaml`: the resolved config used for the run
- `metrics.jsonl`: training, evaluation, and algorithm-specific metrics
- `metrics.png`: a plot of returns and optimization metrics

By default outputs go under `runs/`, which is git-ignored.

## Override Hyperparameters

Use `--set` with dotted config keys:

```sh
uv run python src/train_dqn.py --config configs/cartpole_dqn.yaml --set train.steps=1000 --set seed=456
```

Vectorized DQN/A2C/PPO runs can override `env.num_envs`:

```sh
uv run python src/train_a2c.py --config configs/cartpole_a2c.yaml --set env.num_envs=4
```

For list values, quote the shell argument:

```sh
uv run python src/train_dqn.py --config configs/cartpole_dqn.yaml --set 'model.kwargs.hidden_sizes=[64, 64]'
```

## Replot Existing Metrics

```sh
uv run python src/plot_metrics.py runs/<run-name>/metrics.jsonl
```

## Checks

```sh
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```
