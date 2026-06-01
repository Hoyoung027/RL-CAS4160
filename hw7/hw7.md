# HW7 구현 문서

## 개요

이 문서는 RLHF(Reinforcement Learning from Human Feedback) 구현 과제에서 작성된 코드를 설명한다.
구현 대상 파일은 세 개이며, 각 파일의 TODO 위치와 작성 이유를 기술한다.

---

## 1. `cas4160/networks/reward_predictor.py`

reward model 전체를 담당하는 파일. 관측과 행동을 입력받아 scalar reward를 예측하고,
두 trajectory 간 선호도를 확률로 모델링하며, human feedback을 supervision으로 학습한다.

---

### 1-1. `predict_rewards()` — 보상 예측

**구현 위치:** `predict_rewards()` 내부 `rewards_flat = None` 라인

```python
# TODO: predict reward using `reward_net`
# HINT: you should concatenate observation and action

inp = torch.cat([obs_flat, acts_flat], dim=-1)
rewards_flat = self.reward_net(inp).squeeze(-1)
```

**설명:**

`reward_net`은 `(obs, action)` 쌍을 입력으로 받아 scalar reward를 출력하는 MLP다.
obs와 action을 `dim=-1` 방향으로 concatenate해서 하나의 벡터로 만든 뒤 네트워크에 통과시킨다.

- `obs_flat`: shape `(batch * n_frames, obs_dim)`
- `acts_flat`: shape `(batch * n_frames, act_dim)`
- `inp`: shape `(batch * n_frames, obs_dim + act_dim)`
- `reward_net` 출력은 shape `(batch * n_frames, 1)` → `.squeeze(-1)`으로 `(batch * n_frames,)`로 만든다

이후 `.reshape(batch_size, n_frames)`를 통해 각 trajectory의 timestep별 reward 배열이 된다.

---

### 1-2. `predict_preferences()` — 선호도 확률 예측

**구현 위치:** `predict_preferences()` 내부 None 변수들

```python
# TODO: predict preferences probabiltiy between two trajectories
# HINT: 1. Get reward for each time step using `self.predict_rewards`
#       2. Sum the rewards
#       3. Use Softmax to get probabiltiy

reward_1 = self.predict_rewards(obs1, acts1, training=True)
reward_2 = self.predict_rewards(obs2, acts2, training=True)

reward_sum_1 = reward_1.sum(dim=1, keepdim=True)
reward_sum_2 = reward_2.sum(dim=1, keepdim=True)
pred = F.softmax(torch.cat([reward_sum_1, reward_sum_2], dim=1), dim=1)

return pred
```

**설명:**

과제 수식 (1)을 그대로 구현한 것이다:

$$\hat{P}[\tau_1 \succ \tau_2] = \frac{\exp\left(\sum_t \hat{r}(s_t^1, a_t^1)\right)}{\exp\left(\sum_t \hat{r}(s_t^1, a_t^1)\right) + \exp\left(\sum_t \hat{r}(s_t^2, a_t^2)\right)}$$

각 trajectory의 timestep별 reward를 합산한 뒤, 두 값을 softmax에 통과시킨다.
반환 shape는 `(batch, 2)` — `[:, 0]`이 τ1 선호 확률, `[:, 1]`이 τ2 선호 확률이다.

**`training=True`를 반드시 넘겨야 하는 이유:**

`predict_rewards()` 끝부분에 `if not training: rewards = ptu.to_numpy(rewards)` 가 있다.
`training=False`(기본값)이면 tensor가 numpy로 변환되어 **역전파 그래프가 끊긴다**.
`train_step()`에서 `loss.backward()`를 호출할 때 reward 계산까지 gradient가 흘러야 하므로
학습 중에는 반드시 `training=True`를 지정해야 한다.

---

### 1-3. `compute_loss_and_accuracy()` — loss 계산

**구현 위치:** `compute_loss_and_accuracy()` 내부 `preds = None`, `loss = None` 라인

```python
# TODO: Calculate the reward funciton loss
# HINT: 1. Get preferences
#       2. Calculate loss function

preds = self.predict_preferences(obs1, acts1, obs2, acts2)
loss = -torch.sum(prefs * torch.log(preds + 1e-8), dim=1).mean()
```

**설명:**

과제 수식 (2)의 cross-entropy loss를 구현한 것이다:

$$\mathcal{L}(\hat{r}) = -\sum_{(\tau_1,\tau_2,\mu)\in\mathcal{D}} \left[\mu(1)\log\hat{P}[\tau_1\succ\tau_2] + \mu(2)\log\hat{P}[\tau_2\succ\tau_1]\right]$$

`prefs`는 float scalar (0.0=τ1 선호, 1.0=τ2 선호, 0.5=중립)로 들어오며,
이 함수 위에서 이미 `torch.concat([1 - prefs, prefs], dim=1)`로 `(batch, 2)` 분포 벡터로 변환되어 있다.

- `preds`: shape `(batch, 2)` — 예측된 선호도 확률
- `prefs`: shape `(batch, 2)` — 실제 human label 분포
- `+1e-8`: log(0) 방지

accuracy는 `preds.argmax`와 `prefs.argmax`가 일치하는 비율로, 이미 골격 코드에 작성되어 있다.

---

### 1-4. `train_step()` — 학습 1 스텝

**구현 위치:** `train_step()` 내부 `loss, accuracy = None` 라인 (골격 코드의 Python 문법 오류 포함)

```python
# TODO: Perform the training step

self.optimizer.zero_grad()
loss, accuracy = self.compute_loss_and_accuracy(obs1, acts1, obs2, acts2, prefs)
loss.backward()
self.optimizer.step()
```

**설명:**

표준적인 PyTorch 학습 스텝이다.

1. `zero_grad()`: 이전 스텝의 gradient 초기화
2. `compute_loss_and_accuracy()`: forward pass + loss 계산
3. `loss.backward()`: reward_net 파라미터에 대한 gradient 계산
4. `optimizer.step()`: Adam optimizer로 파라미터 업데이트

골격 코드의 `loss, accuracy = None`은 None을 두 변수로 unpack하는 Python 문법 오류(`TypeError`)이므로
해당 라인 전체를 위 4줄로 교체했다.

---

## 2. `cas4160/agents/pg_agent.py`

PPO agent 파일. reward_predictor를 초기화하는 부분 한 곳만 수정한다.

---

### 2-1. `__init__()` — RewardPredictor input_size 설정

**구현 위치:** `self.reward_predictor = RewardPredictor(input_size=None, ...)` 라인

```python
# TODO: initialize reward predictor
# what should be the input size?
self.reward_predictor = RewardPredictor(
    input_size=ob_dim + ac_dim,
    n_layers=2,
    layer_size=64,
    learning_rate=1e-4
)
```

**설명:**

`reward_net`은 observation과 action을 concatenate한 벡터를 입력으로 받는다.
따라서 input_size는 `ob_dim + ac_dim`이 되어야 한다.
`ob_dim`만 넣으면 action 정보가 빠져 reward 예측이 불완전해지고,
네트워크 첫 번째 Linear layer의 차원이 맞지 않아 런타임 오류가 발생한다.

---

## 3. `cas4160/scripts/run_hw7.py`

학습 루프 전체를 담당하는 파일. preference 수집, reward model 학습, PPO reward 교체를 담당한다.

---

### 3-1. replay buffer에 preference triplet 저장

**구현 위치:** `for annotated_traj in annotated_traj_list:` 블록 내부

```python
# TODO: Store preference triplet to replay buffer
# ...
traj1, traj2, prefs = annotated_traj
replay_buffer.insert(
    observation_1=traj1["observation"],
    observation_2=traj2["observation"],
    action_1=traj1["action"],
    action_2=traj2["action"],
    prefs=np.array([prefs], dtype=np.float32),
)
```

**설명:**

`annotated_traj`는 `(traj1, traj2, float)` 형태의 tuple이다.
`traj["observation"]`과 `traj["action"]`은 `rollout_trajectory()`가 반환하는 dict의 키이며,
각각 shape `(n_frames, obs_dim)`, `(n_frames, act_dim)`의 numpy 배열이다.

`prefs`는 float scalar이므로 `np.array([prefs], dtype=np.float32)`로 shape `(1,)` 배열로 감싸야 한다.
`replay_buffer.insert()`는 내부에서 `(max_size, *prefs.shape)` 형태로 저장하므로,
scalar를 그대로 넣으면 shape 불일치가 발생한다.

---

### 3-2. reward predictor 학습

**구현 위치:** `for _ in range(num_update_reward_predictor):` 블록 내부

```python
# TODO: Train reward predictor
# ...
batch = replay_buffer.sample(batch_size_reward_predictor)
loss, accuracy = agent.reward_predictor.train_step(
    batch["observations_1"], batch["actions_1"],
    batch["observations_2"], batch["actions_2"],
    batch["prefs"],
)
```

**설명:**

replay buffer에서 `batch_size_reward_predictor`(=16)개를 무작위로 샘플링해서
`train_step()`에 넘겨 reward model을 1 스텝 학습시킨다.
이것을 `num_update_reward_predictor`(=100)번 반복함으로써
새로운 human feedback이 들어올 때마다 reward model을 충분히 업데이트한다.

`batch`의 키는 `replay_buffer.sample()`의 반환값 dict 구조 그대로 사용한다:
`observations_1`, `actions_1`, `observations_2`, `actions_2`, `prefs`.

---

### 3-3. train trajectory reward 교체

**구현 위치:** `for traj in trajs:` 블록 내부 (PPO 학습 직전)

```python
# TODO: Replace reward value from reward predictor
# Hint: make sure you set `training=False`
for traj in trajs:
    traj["reward"] = agent.reward_predictor.predict_rewards(
        traj["observation"], traj["action"]
    )
```

**설명:**

PPO는 `traj["reward"]`를 기반으로 Q-value와 advantage를 계산한다.
환경이 제공하는 true reward 대신 학습된 reward model의 예측값으로 교체함으로써
human preference를 반영한 보상 신호로 policy를 학습시키는 것이 RLHF의 핵심이다.

`predict_rewards()`의 `training` 파라미터 기본값이 `False`이므로 별도로 지정하지 않아도 되며,
numpy 배열로 반환되어 `traj["reward"]`의 기존 dtype과 호환된다.

---

### 3-4. eval trajectory reward 교체

**구현 위치:** `for traj in eval_trajs:` 블록 내부 (로깅 직전)

```python
# TODO: Update evaluation reward for evaluation logging
for traj in eval_trajs:
    traj["reward"] = agent.reward_predictor.predict_rewards(
        traj["observation"], traj["action"]
    )
```

**설명:**

TensorBoard에 기록되는 `Eval_AverageReturn`은 `compute_metrics()`가 `traj["reward"].sum()`을 집계한 값이다.
eval trajectory에도 predicted reward를 적용해야 train과 eval의 reward 스케일이 통일되어
학습 곡선이 의미 있는 지표가 된다.

---

## 구현 요약

| 파일 | 함수 | 구현 내용 |
|------|------|-----------|
| `reward_predictor.py` | `predict_rewards()` | obs+act concatenate → MLP → scalar reward |
| `reward_predictor.py` | `predict_preferences()` | reward 합산 → softmax → 선호도 확률 |
| `reward_predictor.py` | `compute_loss_and_accuracy()` | cross-entropy loss (수식 2) |
| `reward_predictor.py` | `train_step()` | zero_grad / backward / step |
| `pg_agent.py` | `__init__()` | `input_size = ob_dim + ac_dim` |
| `run_hw7.py` | 학습 루프 | replay buffer insert |
| `run_hw7.py` | 학습 루프 | reward predictor 학습 (100회) |
| `run_hw7.py` | 학습 루프 | train/eval trajectory reward 교체 |
