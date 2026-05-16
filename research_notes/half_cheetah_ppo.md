# HalfCheetah PPO Report

Date: 2026-05-14; updated 2026-05-17

## Setup

All runs used `src/train_ppo.py`, CPU execution, seed `123`, `HalfCheetah-v5`,
10 evaluation episodes, and config-only changes. The main search compared the
repo default, PPO settings inspired by common HalfCheetah references, a
log-std/parallel-env variant, and rollout-step ablations. The first pass did not
use normalization. The follow-up study added configurable advantage
normalization and enabled observation normalization through the existing
environment wrapper path.

## Best Config

Saved as `configs/half_cheetah_ppo.yaml`.

| Parameter | Value |
|---|---:|
| num envs | 16 |
| observation normalization | enabled, clip=10, epsilon=1e-8 |
| hidden sizes | [256, 256] |
| init log std | -2.0 |
| training steps | 1000000 |
| learning rate | 0.00002 |
| discount factor | 0.98 |
| GAE lambda | 0.92 |
| rollout steps | 128 |
| minibatch size | 64 |
| PPO epochs | 20 |
| clip coef | 0.1 |
| value coef | 0.6 |
| entropy coef | 0.0004 |
| max grad norm | 0.8 |
| normalize advantages | true |

## Result

The current best 1M-step default run is
`runs/half_cheetah_ppo_advnorm_obsnorm_entropy0004_value06_1m_seed123`.

| Run | Steps | Best eval return | Final eval return |
|---|---:|---:|---:|
| adv norm + obs norm, entropy=0.0004, value=0.6 | 1M | 5645.4 +/- 63.6 | 5645.4 +/- 63.6 |
| adv norm + obs norm, entropy=0.0004, value=0.58096 | 1M | 5757.1 +/- 54.9 | 5204.2 +/- 1830.8 |
| adv norm + obs norm, entropy=0.0, value=0.58096 | 801k | 5053.6 +/- 49.2 | 4966.7 +/- 320.1 |
| no adv norm, no obs norm, entropy=0.004, value=0.6 | 1M | 4501.2 +/- 57.7 | 4501.2 +/- 57.7 |
| adv norm only, entropy=0.0004 | 1M | 4160.4 +/- 128.0 | 4160.4 +/- 128.0 |
| adv norm only, entropy=0.004 | 1M | 4040.5 +/- 146.5 | 3997.6 +/- 105.0 |
| adv norm only, entropy=0.0015 | 1M | 3747.0 +/- 29.4 | 3314.6 +/- 1037.6 |

The simplified advantage-normalized default improved the final 1M return by
about `+1144` over the previous no-normalization baseline in this single-seed
comparison. The exact `value_coef=0.58096` run reached the highest intermediate
evaluation, but its final evaluation had very high variance. Rounding the value
coefficient to `0.6` gave a slightly lower peak and a much cleaner final result,
so it is the better default.

## Advantage Normalization Study

Advantage normalization alone was not enough. With the old unnormalized
observation stream, turning on `train.normalize_advantages` made the actor update
scale consistent, but the critic still saw large and spiky targets. The raw
advantage standard deviation and scaled value loss stayed volatile late in
training, and the advantage-normalized runs underperformed the old baseline.

Lowering entropy pressure helped because normalized advantages make the policy
loss live on a smaller, more stable scale. Keeping `entropy_coef=0.004` made the
entropy term relatively too influential, while `0.0004` reduced that pressure
without fully removing exploration. A zero-entropy partial run learned well but
trailed the `0.0004` setting, so the default keeps a small entropy bonus.

Observation normalization was the decisive change. Once observations were
normalized, the critic targets and raw advantages became less erratic, and the
advantage-normalized actor update could outperform the original no-normalization
config. The main practical takeaway is that advantage normalization should be
tuned with the value and observation scale, not treated as an isolated switch.

## Essential PPO Metric

The temporary advantage diagnostics were useful during the ablation, but they are
too noisy for steady-state training logs. Keep only the standard PPO policy-move
diagnostic added during this work:

| Metric | Keep because |
|---|---|
| `approx_kl` | Tracks policy-update size directly and catches overly aggressive normalized updates. |

The debug-only advantage fields can be removed from normal metrics: raw
advantage mean, abs mean, min, max, effective advantage mean, effective advantage
standard deviation, effective advantage abs mean, scaled value loss, and entropy
loss magnitude. Under `normalize_advantages=true`, the effective advantage mean
and standard deviation are mostly expected by construction, so they are good
implementation checks but poor long-term training signals.

## Rollout Ablation

Using the same best config family, longer 5M runs showed that `rollout_steps=128`
remained strongest. Larger rollouts were smoother but learned more slowly.

| Rollout steps | Best eval return | Final eval return | Wall time |
|---:|---:|---:|---:|
| 128 | 5792.9 +/- 54.8 | 5519.6 +/- 423.4 | 58:58 |
| 256 | 5046.7 +/- 73.3 | 4487.4 +/- 1415.0 | 1:03:21 |
| 512 | 4196.9 +/- 27.7 | 4196.9 +/- 27.7 | 59:17 |

Recommendation: keep `rollout_steps=128` as the default. The 512-step setting
has cleaner-looking eval variance, but it underperforms at both 1M and 5M in
this implementation.

## Plots

### Advantage-Normalization Config Search

![Advantage-normalization config search](half_cheetah_ppo_assets/advnorm_config_search.png)

### Value Coefficient 0.6 Check

![Advantage normalization value coefficient 0.6 comparison](half_cheetah_ppo_assets/advnorm_value_coef_06.png)

### Previous No-Normalization 1M Run

![Previous no-normalization 1M metrics](half_cheetah_ppo_assets/simplified_1m_metrics.png)

### 1M Rollout Ablation

![1M rollout ablation](half_cheetah_ppo_assets/rollout_ablation_1m.png)

### 5M Rollout Ablation

![5M rollout ablation](half_cheetah_ppo_assets/rollout_ablation_5m.png)
