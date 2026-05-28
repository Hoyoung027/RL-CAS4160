# HW6: Model-based RL 구현 문서

## 1. 구현 개요

Model-based RL의 두 핵심 요소인 **(1) dynamics model 학습**과 **(2) MPC를 통한 action 선택**을 구현한다.
환경은 HalfCheetah(`cheetah-cas4160-v0`)이며, ensemble of dynamics models + MPC(random shooting / CEM)를 조합한다.

---

## 2. 구현 내용

### 2.1 `run_hw6.py`

#### (1) 초기 데이터 수집 (`itr == 0`)

```python
trajs, envsteps_this_batch = utils.sample_trajectories(
    env, random_policy, config["initial_batch_size"]
)
```

- 첫 iteration에는 학습된 policy가 없으므로 **random policy**로 데이터를 수집한다.
- `sample_trajectories`는 `min_timesteps_per_batch` 이상의 transition이 모일 때까지 trajectory를 반복 수집한다.
- 이 데이터로 dynamics model을 초기 학습하여 이후 on-policy 수집의 기반을 만든다.

#### (2) On-policy 데이터 수집 (`itr > 0`)

```python
trajs, envsteps_this_batch = utils.sample_trajectories(
    env, actor_agent, config["batch_size"]
)
```

- 학습된 `actor_agent`(= `ModelBasedAgent`)의 `get_action()`을 통해 MPC 기반으로 action을 선택한다.
- 모델이 실제로 방문하는 state space를 커버하는 on-policy 데이터를 수집함으로써 dynamics model의 정확도를 높인다.
- random 데이터만으로는 관심 있는 state region을 충분히 커버하지 못하기 때문에 반복적인 on-policy 수집이 필요하다.

#### (3) Ensemble 모델 학습

```python
step_losses = []
for i in range(mb_agent.ensemble_size):
    batch = replay_buffer.sample(config["train_batch_size"])
    loss = mb_agent.get_loss(
        i,
        batch["observations"],
        batch["actions"],
        batch["next_observations"],
    )
    step_losses.append(loss)
```

- ensemble의 각 모델 `i`에 대해 **서로 다른 랜덤 배치**를 샘플링하여 학습한다.
- 같은 배치를 쓰면 ensemble 모델들이 동일한 방향으로 수렴하여 diversity가 사라진다.
- 서로 다른 배치를 씀으로써 각 모델이 독립적인 예측을 하고, ensemble 평균이 분산을 줄이는 효과를 낸다.

---

### 2.2 `model_based_agent.py`

#### (1) `get_loss()` — dynamics model 학습

```python
obs_delta_normalized_hat = self.dynamics_models[i](obs_acs_normalized)
loss = self.loss_fn(obs_delta_normalized_hat, obs_delta_normalized)
```

- 모델 입력: `(obs, acs)`를 concat한 뒤 normalize → `obs_acs_normalized`
- 모델 출력: normalized state difference `δ_normalized` 예측
- 타겟: 실제 `(next_obs - obs)`를 normalize한 값
- **normalize하는 이유**: 타겟 값의 스케일을 통일하여 학습 안정성을 높이고 loss landscape를 개선한다.
- MSELoss로 예측값과 타겟의 L2 거리를 최소화한다.

#### (2) `get_dynamics_predictions()` — next state 예측

```python
obs_delta_normalized = self.dynamics_models[i](obs_acs_normalized)
obs_delta = obs_delta_normalized * self.obs_delta_std + self.obs_delta_mean
pred_next_obs = obs + obs_delta
```

- 모델이 normalized delta를 예측하면 **unnormalize**하여 실제 단위의 delta로 복원한다.
- `s_{t+1} = s_t + unnormalize(f_θ(s_t, a_t))` 공식을 그대로 구현한다.
- `@torch.no_grad()` 데코레이터로 추론 시 gradient 계산을 생략하여 메모리와 속도를 최적화한다.

#### (3) `evaluate_action_sequences()` — action sequence 평가

```python
next_obs = np.stack([
    self.get_dynamics_predictions(i, obs[i], acs)
    for i in range(self.ensemble_size)
])
```

- `obs`는 `(ensemble_size, mpc_num_action_sequences, ob_dim)` 형태로 tile되어 있다.
- 각 ensemble 모델 `i`에 대해 독립적으로 next_obs를 예측하고 stack으로 합친다.
- 이후 각 모델이 예측한 reward를 ensemble 차원에서 평균 내어 불확실성을 줄인다.

#### (4) `get_action()` CEM — Cross-Entropy Method

```python
rewards = self.evaluate_action_sequences(obs, action_sequences)
elite_indices = np.argpartition(rewards, -self.cem_num_elites)[-self.cem_num_elites:]
elite_sequences = action_sequences[elite_indices]

elite_mean = elite_sequences.mean(axis=0)
elite_std = elite_sequences.std(axis=0)

action_sequences = np.random.normal(
    loc=elite_mean,
    scale=elite_std + 1e-6,
    size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
)
action_sequences = np.clip(action_sequences, self.env.action_space.low, self.env.action_space.high)
```

- **CEM 알고리즘 흐름** (`cem_num_iters`회 반복):
  1. 현재 분포에서 K개 action sequence 샘플링 (초기는 uniform)
  2. 각 sequence의 예상 누적 reward 계산
  3. 상위 J개(`cem_num_elites`) elite sequence 선택 (`np.argpartition` 사용)
  4. elite의 mean/std로 diagonal Gaussian을 피팅하여 다음 iteration의 샘플링 분포로 사용
- `np.argpartition`을 쓰는 이유: 완전 정렬보다 O(n) 복잡도로 상위 k개 인덱스를 찾을 수 있다.
- `std + 1e-6`: std가 0이 되는 경우(모든 elite가 동일) numerical stability를 보장한다.
- **MPC 방식**: 매 timestep마다 CEM을 실행하고 최적 sequence의 **첫 번째 action만** 실행한다. 이를 통해 누적 model error를 줄인다.
- 반복 후 최종 mean으로 best action sequence를 구하는 대신, 마지막 iteration의 samples 중 가장 높은 reward를 가진 sequence를 선택한다.

---

## 3. 실험

### 실험 1: Dynamics Model 학습 (`halfcheetah_0_iter`)

dynamics model만 학습하고 policy evaluation은 하지 않는다. model loss가 500 step 이내에 0.2 이하로 떨어지는지 확인한다.

| 설정 | 값 |
|------|----|
| num_iters | 1 |
| initial_batch_size | 20000 (random policy) |
| num_agent_train_steps_per_iter | 500 |
| hidden_size / num_layers | 32 / 1 |
| num_eval_trajectories | 0 (평가 없음) |

```bash
cd /Users/hoyoung/Documents/yonsei/2026-1/강화학습/RL-CAS4160/hw6
nohup python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_0_iter.yaml \
    > logs/exp_0iter.log 2>&1 &
echo "PID: $!"
```

---

### 실험 2: Random Shooting MPC (`halfcheetah_multi_iter`)

on-policy data collection을 포함한 full MBRL 루프. random shooting으로 action을 선택한다. 목표 reward ≥ 300.

| 설정 | 값 |
|------|----|
| num_iters | 15 |
| initial_batch_size | 5000 |
| batch_size | 5000 |
| num_agent_train_steps_per_iter | 1500 |
| mpc_horizon (H) | 15 |
| mpc_num_action_sequences (K) | 1000 |
| ensemble_size | 3 |
| mpc_strategy | random |

```bash
cd /Users/hoyoung/Documents/yonsei/2026-1/강화학습/RL-CAS4160/hw6
nohup python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_multi_iter.yaml \
    > logs/exp_random.log 2>&1 &
echo "PID: $!"
```

---

### 실험 3: CEM MPC (`halfcheetah_cem`)

CEM으로 action을 선택. 목표 reward ≥ 800.

| 설정 | 값 |
|------|----|
| num_iters | 5 |
| mpc_horizon (H) | 15 |
| mpc_num_action_sequences (K) | 1000 (default) |
| cem_num_iters | 4 |
| cem_num_elites | 5 |

```bash
cd /Users/hoyoung/Documents/yonsei/2026-1/강화학습/RL-CAS4160/hw6
nohup python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_cem.yaml \
    > logs/exp_cem.log 2>&1 &
echo "PID: $!"
```

---

### 실험 4: CEM Ablation — H=1 (`halfcheetah_cem_H-1`)

planning horizon을 1로 줄인 경우. 한 step 앞만 내다보기 때문에 장기 보상을 고려하지 못해 성능이 저하될 것으로 예상한다.

| 설정 | 값 |
|------|----|
| mpc_horizon (H) | **1** |
| 나머지 | CEM 기본값과 동일 |

```bash
cd /Users/hoyoung/Documents/yonsei/2026-1/강화학습/RL-CAS4160/hw6
nohup python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_cem_H-1.yaml \
    > logs/exp_cem_H1.log 2>&1 &
echo "PID: $!"
```

---

### 실험 5: CEM Ablation — K=50 (`halfcheetah_cem_K-50`)

action sequence 후보 수를 1000에서 50으로 줄인 경우. 탐색 공간이 작아져 최적 action을 찾기 어려워지므로 성능이 저하될 것으로 예상한다.

| 설정 | 값 |
|------|----|
| mpc_num_action_sequences (K) | **50** |
| 나머지 | CEM 기본값과 동일 |

```bash
cd /Users/hoyoung/Documents/yonsei/2026-1/강화학습/RL-CAS4160/hw6
nohup python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_cem_K-50.yaml \
    > logs/exp_cem_K50.log 2>&1 &
echo "PID: $!"
```

---

## 4. 전체 실험 한꺼번에 실행

```bash
cd /Users/hoyoung/Documents/yonsei/2026-1/강화학습/RL-CAS4160/hw6
mkdir -p logs

nohup python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_0_iter.yaml \
    > logs/exp_0iter.log 2>&1 &
echo "exp_0iter PID: $!"

nohup python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_multi_iter.yaml \
    > logs/exp_random.log 2>&1 &
echo "exp_random PID: $!"

nohup python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_cem.yaml \
    > logs/exp_cem.log 2>&1 &
echo "exp_cem PID: $!"

nohup python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_cem_H-1.yaml \
    > logs/exp_cem_H1.log 2>&1 &
echo "exp_cem_H1 PID: $!"

nohup python cas4160/scripts/run_hw6.py -cfg experiments/mpc/halfcheetah_cem_K-50.yaml \
    > logs/exp_cem_K50.log 2>&1 &
echo "exp_cem_K50 PID: $!"
```

로그 실시간 확인:
```bash
tail -f logs/exp_cem.log
```

프로세스 확인:
```bash
ps aux | grep run_hw6
```
