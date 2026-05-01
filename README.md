# learn-rl

Small DQN experiments for Gymnasium tasks.

## Train CartPole DQN

```sh
uv run python src/train_cartpole.py
```

The default experiment config lives at `configs/cartpole_dqn.yaml`. Each run writes:

- `config.yaml`: the resolved config used for the run
- `metrics.jsonl`: step-wise training, evaluation, and epsilon metrics
- `metrics.png`: a plot of returns, TD loss, and epsilon

By default outputs go under `runs/`, which is git-ignored.

## Override Hyperparameters

Use `--set` with dotted config keys:

```sh
uv run python src/train_cartpole.py --set train.steps=1000 --set seed=456
```

For list values, quote the shell argument:

```sh
uv run python src/train_cartpole.py --set 'model.hidden_sizes=[64, 64]'
```

## Replot Existing Metrics

```sh
uv run python src/plot_metrics.py runs/<run-name>/metrics.jsonl
```
