# Observation Normalization and MLP Architecture

Date: 2026-05-17

This note consolidates the observation-normalization ablations and the follow-up MLP architecture experiments.

## Summary

Observation normalization was not a simple win or loss. It helped some tasks, but with the original small ReLU MLP it also created severe failures. The important finding is that the wrapper itself was not the main issue. Observation normalization changed the input geometry, and that exposed a representation problem in the policy/value network.

The old actor-critic MLP used plain ReLU. With normalized observations, Acrobot A2C and PPO collapsed to the timeout floor. SB3 did not collapse with its default MLP, but SB3 did collapse when forced to use our old ReLU/non-orthogonal architecture. That made the diagnosis concrete: observation normalization was interacting badly with the model architecture.

The current MLP recipe is:

```yaml
model:
  kwargs:
    activation: relu
    layer_norm: true
    orthogonal_init: false
```

LayerNorm is the key part. It keeps hidden preactivation scale controlled before the ReLU gate, so ReLU no longer behaves like a brittle one-sided filter after observation normalization recenters the input distribution.

## Observation-Only Normalization

The wrapper normalizes observations using running mean and variance. Evaluation envs share the training `RunningMeanStd` and do not update it. This matches the broad VecNormalize pattern, but the local wrapper is observation-only: it does not normalize rewards.

That distinction matters. RL Zoo's `normalize: true` usually means SB3 `VecNormalize` behavior, where reward normalization may also be active. RL Zoo also relies on details this repo did not always expose during the first ablations: Tanh actor-critic MLPs, orthogonal initialization, advantage normalization, schedules, optimizer choices, and gSDE for some continuous tasks.

### Continuous Control

Run root: `runs/obs_norm_matrix_20260514`

| Task | Algo | Raw final | Obs-norm final | Delta |
|---|---|---:|---:|---:|
| Pendulum-v1 | A2C | `-1029.5` | `-1250.3` | `-220.7` |
| Pendulum-v1 | PPO | `-117.9` | `-580.8` | `-462.9` |
| MountainCarContinuous-v0 | A2C | `-0.8` | `93.4` | `+94.2` |
| MountainCarContinuous-v0 | PPO | `-0.1` | `-0.9` | `-0.9` |
| LunarLanderContinuous-v3 | A2C | `-16.6` | `-82.1` | `-65.4` |
| LunarLanderContinuous-v3 | PPO | `269.8` | `277.1` | `+7.3` |

The large positive result was MountainCarContinuous A2C. LunarLanderContinuous PPO improved slightly. Pendulum and LunarLanderContinuous A2C got worse.

The key lesson was not "do not normalize observations." The key lesson was that observation normalization can destabilize a model that was tuned around raw observation scale.

### Discrete Control

Run roots:

- `runs/obs_norm_discrete_matrix_20260514`
- `runs/obs_norm_discrete_rl_zoo_matrix_20260515`

The first matrix used local configs. The RL-Zoo-mapped matrix was the more important check because it used settings closer to common SB3/RL Zoo practice.

RL-Zoo-mapped final returns:

| Task | Algo | Raw final | Obs-norm final | Delta |
|---|---|---:|---:|---:|
| CartPole-v1 | A2C | `500.0` | `9.4` | `-490.6` |
| CartPole-v1 | PPO | `500.0` | `500.0` | `+0.0` |
| MountainCar-v0 | A2C | `-200.0` | `-200.0` | `+0.0` |
| MountainCar-v0 | PPO | `-200.0` | `-200.0` | `+0.0` |
| Acrobot-v1 | A2C | `-80.0` | `-500.0` | `-420.0` |
| Acrobot-v1 | PPO | `-74.2` | `-500.0` | `-425.8` |
| LunarLander-v3 | A2C | `54.4` | `-103.9` | `-158.3` |
| LunarLander-v3 | PPO | `273.9` | `269.8` | `-4.1` |

This looked bad for observation normalization, especially Acrobot and CartPole A2C. But the same comparison also revealed the likely cause: the normalized runs were not failing like SB3 defaults. They were failing like a specific MLP architecture.

## Architecture Diagnosis

Observation normalization centers observations around zero. That is exactly where a plain ReLU network can become fragile in on-policy RL.

In supervised learning, a weak ReLU representation can often recover because the data distribution is fixed. In on-policy RL, the policy creates the next data distribution. If early hidden gates produce poor actions, the rollout data narrows around those poor actions, and the model trains on the consequences of its own representation failure.

Tanh avoids part of this because it preserves signed information around zero. ReLU + LayerNorm solves it differently: keep ReLU, but normalize each hidden vector before applying the gate.

### Acrobot Failure Mode

On Acrobot with observation normalization:

| Algo | Plain ReLU | Tanh | ReLU + LayerNorm |
|---|---:|---:|---:|
| A2C | `-500.0 +/- 0.0` | `-82.4 +/- 7.5` | `-78.3 +/- 6.7` |
| PPO | `-500.0 +/- 0.0` | `-69.4 +/- 4.2` | `-75.4 +/- 10.0` |

This was the turning point. The normalized environment was usable; the original ReLU representation was not.

## ReLU + LayerNorm vs Tanh

Run root: `runs/activation_layernorm_global_ablation_20260516`

This ablation kept observation normalization on and compared Tanh against ReLU + LayerNorm across the A2C/PPO classic-control matrix. Orthogonal initialization was fixed to true in this first architecture comparison to isolate the activation/normalization change.

| Task | Algo | Tanh | ReLU + LayerNorm | Winner |
|---|---|---:|---:|---|
| CartPole-v1 | A2C | `500.0` | `500.0` | Tie |
| CartPole-v1 | PPO | `500.0` | `500.0` | Tie |
| MountainCar-v0 | A2C | `-113.0` | `-137.1` | Tanh |
| MountainCar-v0 | PPO | `-109.9` | `-101.0` | ReLU+LN |
| Acrobot-v1 | A2C | `-84.2` | `-78.3` | ReLU+LN |
| Acrobot-v1 | PPO | `-72.6` | `-75.4` | Tanh |
| LunarLander-v3 | A2C | `-56.8` | `99.0` | ReLU+LN |
| LunarLander-v3 | PPO | `243.2` | `276.1` | ReLU+LN |
| Pendulum-v1 | A2C | `-1179.0` | `-825.7` | ReLU+LN |
| Pendulum-v1 | PPO | `-1436.3` | `-873.9` | ReLU+LN |
| MountainCarContinuous-v0 | A2C | `95.6` | `90.6` | Tanh |
| MountainCarContinuous-v0 | PPO | `-0.5` | `-0.1` | ReLU+LN |
| LunarLanderContinuous-v3 | A2C | `-290.6` | `143.0` | ReLU+LN |
| LunarLanderContinuous-v3 | PPO | `256.7` | `273.8` | ReLU+LN |

Win count:

| Architecture | Count |
|---|---:|
| ReLU + LayerNorm | 9 |
| Tanh | 3 |
| Tie | 2 |

The largest gains came from tasks where normalized observations exposed instability:

- LunarLanderContinuous A2C: `-290.6` to `143.0`
- Pendulum PPO: `-1436.3` to `-873.9`
- LunarLander A2C: `-56.8` to `99.0`
- LunarLander PPO: `243.2` to `276.1`

Tanh remained competitive and faster, but ReLU + LayerNorm was more robust across this matrix.

## Orthogonal Initialization

Run root: `runs/relu_layernorm_orthogonal_ablation_20260516`

After selecting ReLU + LayerNorm, I tested whether orthogonal initialization should remain part of the default recipe. It did not produce a global win.

| Initialization | Count |
|---|---:|
| Orthogonal init true | 6 |
| Orthogonal init false | 6 |
| Tie | 2 |

Important rows:

| Task | Algo | Orth false | Orth true | Better |
|---|---|---:|---:|---|
| Acrobot-v1 | A2C | `-92.1` | `-78.3` | Orth true |
| LunarLander-v3 | A2C | `233.0` | `99.0` | Orth false |
| Pendulum-v1 | A2C | `-298.7` | `-825.7` | Orth false |
| Pendulum-v1 | PPO | `-230.1` | `-873.9` | Orth false |
| LunarLanderContinuous-v3 | PPO | `142.2` | `273.8` | Orth true |

The interpretation is that LayerNorm already controls hidden activation scale. Orthogonal initialization still changes early policy dynamics, but that effect is task-sensitive. Since the global signal was neutral and some negative deltas were large, the checked-in MLP configs use `orthogonal_init: false`.

## Cost

LayerNorm is not expensive because of parameter count. For a 2x64 Acrobot actor-critic MLP:

| Model | Parameters |
|---|---:|
| Tanh | `9,476` |
| ReLU + LayerNorm | `9,988` |
| Extra | `+512` |

The cost is runtime. LayerNorm adds reduction-heavy CPU work on small rollout batches. In profiler runs on Acrobot, `native_layer_norm` and `native_layer_norm_backward` dominated ReLU + LayerNorm training time for batch 16. In the global matrix, ReLU + LayerNorm was roughly:

| Algorithm | Slowdown |
|---|---:|
| A2C | `~2.0x` |
| PPO | `~1.5x` |
| Overall | `~1.7x` |

That is the tradeoff: ReLU + LayerNorm is more robust, but it is slower on this CPU-heavy small-MLP workload.

## Current Decision

Observation normalization stays available as an env option. It is not automatically enabled everywhere, because its value depends on task, reward scale, and model architecture.

The default MLP architecture in configs now uses ReLU + LayerNorm with orthogonal init off:

```yaml
activation: relu
layer_norm: true
orthogonal_init: false
```

This is not a claim that Tanh is obsolete. Tanh is still a strong and simple baseline for classic-control actor-critic MLPs. The reason to prefer ReLU + LayerNorm here is pragmatic: it fixed the collapse created by observation normalization and won the single-seed global architecture matrix.

Remaining work:

- Run multi-seed confirmation before treating these as benchmark results.
- Add reward normalization before comparing directly against RL Zoo `normalize: true`.
- Revisit PPO advantage normalization and learning-rate schedules, since both are part of the usual SB3/RL Zoo recipe.

## Sources

- [RL Baselines3 Zoo A2C hyperparameters](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml)
- [RL Baselines3 Zoo PPO hyperparameters](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml)
- [RL Baselines3 Zoo training guide](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/train.html)
- [Stable Baselines3 VecNormalize documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
