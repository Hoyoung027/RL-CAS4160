# HW2 구현 노트

---

## 강화학습 전체 학습 플로우

### 1단계: Trajectory 수집

**Trajectory** = 에이전트가 환경과 상호작용한 하나의 기록

```
trajectory 1: (s0,a0,r0) → (s1,a1,r1) → (s2,a2,r2) → 종료
trajectory 2: (s0,a0,r0) → (s1,a1,r1) → 종료
trajectory 3: (s0,a0,r0) → ... → 종료
```

현재 policy(actor)로 여러 trajectory를 실행해서 데이터를 모음.

---

### 2단계: Q값 계산

각 trajectory의 보상으로 각 시점의 Q값 계산:

```
trajectory 1의 보상: [10, 20, 30]

Q(s0) = 10 + γ*20 + γ²*30
Q(s1) =      20   + γ*30
Q(s2) =             30
```

---

### 3단계: 배치(Batch) 구성

모든 trajectory의 데이터를 **하나의 큰 배열로 합침**:

```
trajectory 1: 3개 스텝
trajectory 2: 2개 스텝
trajectory 3: 5개 스텝
                 ↓ np.concatenate
배치 크기 = 10개 (s, a, Q값 각각 10개짜리 배열)
```

---

### 4단계: Advantage 계산

배치 전체에 대해 한번에 계산:

```
V(s) = Critic이 추정한 값  ← 신경망에 obs 10개 한번에 넣음
Advantage = Q값 - V(s)      ← 10개짜리 배열
```

---

### 5단계: Actor 업데이트

10개의 (s, a, advantage) 샘플을 **한번에** 넣어서 학습:

```
Loss = -mean(advantage * log π(a|s))
```

advantage가 양수인 행동 → 확률 높이기  
advantage가 음수인 행동 → 확률 낮추기

---

### 6단계: Critic 업데이트

10개의 (s, Q값) 샘플을 **한번에** 넣어서 학습:

```
Loss = mean((V(s) - Q값)²)
```

V(s) 예측이 Q값에 가까워지도록 신경망 가중치 업데이트.

---

### 전체 루프

```
┌─────────────────────────────────────┐
│  현재 policy로 trajectory N개 수집   │
│           ↓                         │
│  Q값 계산 (Monte Carlo)              │
│           ↓                         │
│  배치로 합치기                        │
│           ↓                         │
│  Advantage 계산 (Q - V)             │
│           ↓                         │
│  Actor 업데이트 (더 좋은 행동 강화)   │
│           ↓                         │
│  Critic 업데이트 (V 추정 개선)        │
└──────────────── 반복 ───────────────┘
```

---

## 1. `cas4160/infrastructure/utils.py` — 환경 롤아웃

### 역할

Policy를 실제 환경에 실행(롤아웃)해서 학습 데이터 `(s, a, r, s', done)`을 수집하는 부분이다. 이 데이터가 없으면 policy gradient 자체가 시작되지 않는다.

---

### 구현 코드 (`rollout_trajectory`)

```python
def rollout_trajectory(
    env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob, _ = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render an image
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render()
            image_obs.append(img)

        assert ob.ndim == 1
        # TODO use the most recent ob and the policy to decide what to do
        ac: np.ndarray = policy.get_action(ob)

        # check if output action matches the action space
        assert ac.shape == env.action_space.shape

        # TODO: use that action to take a step in the environment
        next_ob, rew, terminated, truncated, _ = env.step(ac)

        # TODO rollout can end due to (terminated or truncated), or due to max_length
        steps += 1
        rollout_done: bool = terminated or truncated or (steps >= max_length)

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }
```

---

### 각 TODO 설명

#### TODO 1: `ac = policy.get_action(ob)`

```python
ac: np.ndarray = policy.get_action(ob)
```

현재 관측값 `ob`를 policy network에 넣어서 action을 샘플링한다.

`get_action()`은 내부적으로:
1. `ob`를 torch tensor로 변환
2. `forward(obs)` 호출 → 확률 분포 반환 (Categorical 또는 Normal)
3. 그 분포에서 `.sample()` → action 추출
4. numpy array로 변환해서 반환

강의에서 policy gradient는 `π_θ(a|s)`에서 action을 샘플링한다고 했는데, 그것을 실제 코드로 하는 부분이 바로 이 한 줄이다.

#### TODO 2: `next_ob, rew, terminated, truncated, _ = env.step(ac)`

```python
next_ob, rew, terminated, truncated, _ = env.step(ac)
```

선택한 action을 환경에 적용해서 다음 상태, 보상, 종료 여부를 받아온다. Gymnasium API 기준으로 `env.step()`은 5개를 반환한다:

| 반환값 | 의미 |
|--------|------|
| `next_ob` | 다음 상태 s' |
| `rew` | 보상 r(s, a) |
| `terminated` | MDP가 자연종료 (목표 달성 또는 실패) |
| `truncated` | 시간 제한 등으로 강제종료 |
| `_` | info dict (사용 안 함) |

이렇게 얻은 `(ob, ac, rew, next_ob)` 튜플이 강의에서의 `(s_t, a_t, r_t, s_{t+1})`에 해당한다.

#### TODO 3: `rollout_done = terminated or truncated or (steps >= max_length)`

```python
steps += 1
rollout_done: bool = terminated or truncated or (steps >= max_length)
```

롤아웃이 끝나는 조건은 세 가지다:

- `terminated`: 환경이 자연스럽게 끝남 (e.g., CartPole이 넘어짐)
- `truncated`: 시간 제한 초과로 강제 종료
- `steps >= max_length`: 우리가 설정한 최대 길이 도달

`terminals` 배열에 이 값을 저장하는 이유는, 나중에 GAE(Generalized Advantage Estimation)를 계산할 때 에피소드 경계를 알아야 하기 때문이다. 강의 슬라이드에서 GAE 공식의 `(1 - terminal_t)` 마스크가 바로 이것을 활용한다:

```
A_t = δ_t + (γ·λ) · (1 - terminal_t) · A_{t+1}
```

에피소드가 끝난 시점 이후의 가치는 0이어야 하므로, `terminal=True`이면 다음 스텝의 advantage를 더하지 않는다.

---

### 전체 흐름 요약

```
env.reset()
   ↓
ob (현재 상태)
   ↓
policy.get_action(ob)   ← TODO 1: π_θ(a|s)에서 샘플링
   ↓
ac (선택한 행동)
   ↓
env.step(ac)            ← TODO 2: 환경에 행동 적용
   ↓
next_ob, rew, terminated, truncated
   ↓
rollout_done 계산        ← TODO 3: 종료 조건 판단
   ↓
(ob, ac, rew, next_ob, rollout_done) 저장
   ↓
ob = next_ob (다음 스텝으로)
   ↓
rollout_done이면 break, 아니면 반복
```

---

## 2. `cas4160/networks/critics.py` — 가치함수 (Critic)

### 역할

Critic은 상태 `s`를 입력받아 그 상태의 가치 `V^π(s)`를 예측하는 신경망이다. 강의에서 말한 "baseline"이 바로 이 `V^π(s)`이다. Policy gradient의 분산을 줄이기 위해 advantage `A = Q(s,a) - V(s)`를 계산할 때 사용된다.

---

### 구현 코드 (`ValueCritic`)

```python
class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        return self.network(obs).squeeze(-1)

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        assert obs.ndim == 2
        assert q_values.ndim == 1

        # TODO: update the critic using the observations and q_values
        predicted_values = self(obs)
        loss = F.mse_loss(predicted_values, q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Baseline Loss": ptu.to_numpy(loss),
        }
```

---

### 각 TODO 설명

#### TODO 1: `forward` — `self.network(obs).squeeze(-1)`

```python
return self.network(obs).squeeze(-1)
```

`ptu.build_mlp()`는 `output_size=1`로 만들어져 있어서 출력 shape이 `(batch, 1)`이다. 그런데 이후 advantage 계산에서 q_values는 `(batch,)` 형태의 1D 배열이다. 두 텐서의 shape이 맞지 않으면 loss 계산에서 broadcasting 오류가 생기므로 `.squeeze(-1)`로 마지막 차원을 제거해 `(batch,)`로 맞춰준다.

```
network 출력: (batch, 1)  →  squeeze(-1)  →  (batch,)
q_values:     (batch,)
```

#### TODO 2: `update` — MSE loss + backprop

```python
predicted_values = self(obs)
loss = F.mse_loss(predicted_values, q_values)

self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

Critic을 학습시키는 방법은 지도학습의 회귀(regression)와 동일하다. 강의 슬라이드의 수식:

```
L(ϕ) = (1/2) * Σ ‖V^π_ϕ(s^i) - y_i‖²
```

여기서 target `y_i`가 바로 `q_values`이다. `q_values`는 pg_agent에서 계산된 실제 discounted reward-to-go이고, critic은 이것을 예측하도록 MSE loss로 학습된다.

업데이트 3단계:
1. `optimizer.zero_grad()` — 이전 gradient 초기화 (PyTorch는 gradient를 누적하므로 반드시 필요)
2. `loss.backward()` — 역전파로 gradient 계산
3. `optimizer.step()` — Adam optimizer로 파라미터 업데이트

---

### 왜 Critic이 필요한가?

강의에서 policy gradient의 weight로 사용할 수 있는 것들을 비교했다:

| weight | 특징 |
|--------|------|
| `r(τ)` (전체 보상) | 분산 매우 큼 |
| `Σ r(s_{t'}, a_{t'})` (reward-to-go) | 분산 줄어듦 |
| `Q^π(s,a)` (Q-value) | 분산 더 줄어듦 |
| `A^π(s,a) = Q - V` (advantage) | **분산 가장 작음** ← Critic 필요 |

Critic `V^π_ϕ(s)`를 학습해두면 advantage `A = Q(s,a) - V(s)`를 계산할 수 있고, 이것으로 policy를 업데이트하면 gradient의 분산이 크게 줄어든다. "Actor-critic" 이름의 유래가 바로 여기다: Actor(policy)는 Critic(value function)의 평가를 받아 학습한다.

---

## 3. `cas4160/networks/policies.py` — 정책 (Actor)

### 역할

실제로 action을 선택하는 policy 신경망이다. 관측값 `s`를 받아 확률 분포를 출력하고, 그 분포에서 action을 샘플링한다. Policy gradient로 학습되며, PPO에서는 clipped objective로 업데이트된다.

---

### 구현 코드 (`MLPPolicy` + `MLPPolicyPG`)

```python
class MLPPolicy(nn.Module):

    def __init__(self, ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate):
        super().__init__()
        if discrete:
            self.logits_net = ptu.build_mlp(ob_dim, ac_dim, n_layers, layer_size).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(ob_dim, ac_dim, n_layers, layer_size).to(ptu.device)
            self.logstd = nn.Parameter(torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device))
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())
        self.optimizer = optim.Adam(parameters, learning_rate)
        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        obs_tensor = ptu.from_numpy(obs[None])  # (1, ob_dim)
        dist = self(obs_tensor)
        action = ptu.to_numpy(dist.sample()[0])
        return action

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            # HINT: use torch.distributions.Categorical to define the distribution.
            logits = self.logits_net(obs)
            dist = distributions.Categorical(logits=logits)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            # HINT: use torch.distributions.Normal to define the distribution.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            dist = distributions.Normal(mean, std)
        return dist


class MLPPolicyPG(MLPPolicy):

    def update(self, obs, actions, advantages) -> dict:
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        # HINT: don't forget to do `self.optimizer.step()`!
        dist = self(obs)
        log_prob = dist.log_prob(actions)
        if not self.discrete:
            # continuous: log_prob shape은 (batch, ac_dim) → 각 action dim의 log_prob 합산
            log_prob = log_prob.sum(axis=-1)
        loss = -(log_prob * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"Actor Loss": ptu.to_numpy(loss)}

    def ppo_update(self, obs, actions, advantages, old_logp, ppo_cliprange=0.2) -> dict:
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        old_logp = ptu.from_numpy(old_logp)

        # TODO: Implement the ppo update.
        # HINT: calculate logp first, and then caculate ratio and clipped loss.
        # HINT: ratio is the exponential of the difference between logp and old_logp.
        # HINT: You can use torch.clamp to clip values.
        dist = self(obs)
        logp = dist.log_prob(actions)
        if not self.discrete:
            logp = logp.sum(axis=-1)

        ratio = torch.exp(logp - old_logp)
        clipped_ratio = torch.clamp(ratio, 1 - ppo_cliprange, 1 + ppo_cliprange)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"PPO Loss": ptu.to_numpy(loss)}
```

---

### 각 TODO 설명

#### TODO 1: `get_action` — 단일 관측값으로 action 샘플링

```python
obs_tensor = ptu.from_numpy(obs[None])  # (1, ob_dim)
dist = self(obs_tensor)
action = ptu.to_numpy(dist.sample()[0])
```

`obs`는 shape `(ob_dim,)`인 1D numpy array이다. network는 batch 처리를 기대하므로 `obs[None]`으로 `(1, ob_dim)`으로 만들어 넣는다. `forward()`가 분포를 반환하면 `.sample()`로 action을 하나 뽑고, `[0]`으로 batch 차원을 제거한 뒤 numpy로 변환한다.

`@torch.no_grad()` 데코레이터가 붙어있는 이유는 rollout 중 action을 선택할 때는 gradient를 추적할 필요가 없어서 메모리와 연산을 절약하기 위해서다.

#### TODO 2: `forward` (discrete) — Categorical 분포

```python
logits = self.logits_net(obs)
dist = distributions.Categorical(logits=logits)
```

Discrete action space (예: CartPole)에서는 각 action에 대한 unnormalized score(logits)를 출력하고, `Categorical` 분포로 감싼다. 내부적으로 softmax를 적용해 확률로 변환하므로 직접 softmax를 쓸 필요 없다.

#### TODO 3: `forward` (continuous) — Normal 분포

```python
mean = self.mean_net(obs)
std = torch.exp(self.logstd)
dist = distributions.Normal(mean, std)
```

Continuous action space (예: HalfCheetah)에서는 각 action 차원의 평균을 network로 예측하고, 표준편차는 학습 가능한 파라미터 `logstd`로 관리한다. `logstd`를 직접 학습하는 이유는 std가 항상 양수여야 하는데, log 공간에서 학습하면 이 제약을 자동으로 만족하기 때문이다 (`exp`는 항상 양수).

#### TODO 4: `update` — Policy gradient loss

```python
dist = self(obs)
log_prob = dist.log_prob(actions)
if not self.discrete:
    log_prob = log_prob.sum(axis=-1)
loss = -(log_prob * advantages).mean()
```

강의의 policy gradient 수식을 그대로 구현한다:

```
∇J(θ) = E[ ∇log π_θ(a|s) · A(s,a) ]
```

loss에 음수를 붙이는 이유는 PyTorch optimizer가 기본적으로 **최소화**를 하는데, 우리는 J(θ)를 **최대화**해야 하기 때문이다.

continuous의 경우 `Normal.log_prob(actions)`은 shape `(batch, ac_dim)`이다. action의 각 차원이 독립적이라고 가정하면 joint log probability는 각 차원의 합이므로 `.sum(axis=-1)`로 `(batch,)` 형태로 합산한다.

#### TODO 5: `ppo_update` — PPO clipped loss

```python
dist = self(obs)
logp = dist.log_prob(actions)
if not self.discrete:
    logp = logp.sum(axis=-1)

ratio = torch.exp(logp - old_logp)
clipped_ratio = torch.clamp(ratio, 1 - ppo_cliprange, 1 + ppo_cliprange)
loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
```

강의의 PPO objective:

```
J(θ) = E[ min( ratio·A, clip(ratio, 1-ε, 1+ε)·A ) ]
```

`ratio = π_θ(a|s) / π_old(a|s)` 를 log 공간에서 계산하면 수치 안정성이 높아진다:

```
ratio = exp(log π_θ - log π_old)
```

`torch.clamp`로 ratio를 `[1-ε, 1+ε]` 범위로 제한한 뒤, 원래 ratio와 clipped ratio 중 작은 값을 취한다. 이렇게 하면:
- advantage > 0일 때: ratio가 너무 커지면(policy가 너무 많이 바뀌면) clamp로 제한 → 과도한 업데이트 방지
- advantage < 0일 때: ratio가 너무 작아지면 clamp로 제한 → 마찬가지로 과도한 업데이트 방지

---

### Discrete vs Continuous 비교

| | Discrete (e.g. CartPole) | Continuous (e.g. HalfCheetah) |
|--|--|--|
| 네트워크 출력 | logits `(batch, ac_dim)` | mean `(batch, ac_dim)` |
| 분포 | `Categorical(logits)` | `Normal(mean, std)` |
| log_prob shape | `(batch,)` | `(batch, ac_dim)` → `.sum(-1)` 필요 |
| std | 없음 | 학습 가능한 `logstd` 파라미터 |

---

## 4. `cas4160/agents/pg_agent.py` — 핵심 학습 로직

### 역할

Policy Gradient의 전체 학습 루프를 담당한다. 수집된 trajectory 데이터로부터 Q값을 계산하고, advantage를 추정하고, actor와 critic을 업데이트하는 모든 과정이 여기에 있다.

---

### 구현 코드

```python
def update(self, obs, actions, rewards, terminals) -> dict:
    # step 1: Q값 계산 (이미 제공됨)
    q_values = self._calculate_q_vals(rewards)

    # TODO: flatten
    obs = np.concatenate(obs)
    actions = np.concatenate(actions)
    rewards = np.concatenate(rewards)
    terminals = np.concatenate(terminals)
    q_values = np.concatenate(q_values)

    # step 2: advantage 추정
    advantages = self._estimate_advantage(obs, rewards, q_values, terminals)

    if not self.use_ppo:
        # TODO: normalize
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # TODO: actor 업데이트
        info = self.actor.update(obs, actions, advantages)

        if self.critic is not None:
            # TODO: critic 업데이트
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)
            info.update(critic_info)
    else:
        logp = self._calculate_log_probs(obs, actions)
        n_batch = len(obs)
        inds = np.arange(n_batch)
        for _ in range(self.n_ppo_epochs):
            np.random.shuffle(inds)
            minibatch_size = (n_batch + (self.n_ppo_minibatches - 1)) // self.n_ppo_minibatches
            for start in range(0, n_batch, minibatch_size):
                end = start + minibatch_size
                obs_slice, actions_slice, advantages_slice, logp_slice = (
                    arr[inds[start:end]] for arr in (obs, actions, advantages, logp)
                )
                # TODO: normalize slice
                if self.normalize_advantages:
                    advantages_slice = (advantages_slice - advantages_slice.mean()) / (advantages_slice.std() + 1e-8)

                # TODO: PPO 업데이트
                info = self.actor.ppo_update(obs_slice, actions_slice, advantages_slice, logp_slice, self.ppo_cliprange)

        for _ in range(self.baseline_gradient_steps):
            critic_info = self.critic.update(obs, q_values)
        info.update(critic_info)

    return info


def _calculate_q_vals(self, rewards):
    if not self.use_reward_to_go:
        # TODO: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}  (전체 궤적, 모든 t 동일)
        q_values = [self._discounted_return(r) for r in rewards]
    else:
        # TODO: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) r_{t'}  (t 이후만)
        q_values = [self._discounted_reward_to_go(r) for r in rewards]
    return q_values


def _estimate_advantage(self, obs, rewards, q_values, terminals):
    if self.critic is None:
        # TODO: baseline 없으면 advantage = Q값 그대로
        advantages = q_values
    else:
        # TODO: critic으로 V(s) 계산
        values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)))

        if self.gae_lambda is None:
            # TODO: A = Q - V
            advantages = q_values - values
        else:
            # TODO: GAE
            batch_size = obs.shape[0]
            values = np.append(values, [0])       # 더미 V(s_{T+1}) = 0
            advantages = np.zeros(batch_size + 1)

            for i in reversed(range(batch_size)):
                delta = rewards[i] + self.gamma * (1 - terminals[i]) * values[i + 1] - values[i]
                advantages[i] = delta + self.gamma * self.gae_lambda * (1 - terminals[i]) * advantages[i + 1]

            advantages = advantages[:-1]  # 더미 제거
    return advantages


def _discounted_return(self, rewards):
    # TODO: 전체 합산, 모든 t에 동일한 값 반환
    T = len(rewards)
    gammas = np.array([self.gamma ** t for t in range(T)])
    total = np.sum(gammas * rewards)
    ret = np.full(T, total, dtype=np.float32)
    return ret


def _discounted_reward_to_go(self, rewards):
    # TODO: t 시점부터 역방향으로 누적 합산
    T = len(rewards)
    ret = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + self.gamma * running
        ret[t] = running
    return ret


def _calculate_log_probs(self, obs, actions):
    # TODO: 현재 policy로 log π(a|s) 계산
    with torch.no_grad():
        obs_tensor = ptu.from_numpy(obs)
        actions_tensor = ptu.from_numpy(actions)
        dist = self.actor(obs_tensor)
        logp = dist.log_prob(actions_tensor)
        if not self.actor.discrete:
            logp = logp.sum(axis=-1)
        logp = ptu.to_numpy(logp)
    return logp
```

---

### 각 TODO 설명

#### `update` — flatten

```python
obs = np.concatenate(obs)
actions = np.concatenate(actions)
rewards = np.concatenate(rewards)
terminals = np.concatenate(terminals)
q_values = np.concatenate(q_values)
```

입력으로 들어오는 `obs`, `actions` 등은 trajectory별 배열의 리스트다. 예를 들어 3개의 trajectory가 있고 각 길이가 100이면, `obs`는 `[array(100,4), array(100,4), array(100,4)]`이다. `np.concatenate`로 `(300, 4)` 하나로 합쳐야 이후 연산을 배치로 처리할 수 있다.

#### `_discounted_return` — 전체 궤적 할인 보상

```python
gammas = np.array([self.gamma ** t for t in range(T)])
total = np.sum(gammas * rewards)
ret = np.full(T, total, dtype=np.float32)
```

수식: `Q(s_t, a_t) = Σ_{t'=0}^T γ^{t'} r_{t'}`

모든 timestep에서 동일한 값을 사용한다 (causality를 무시한 naive PG). `np.full`로 같은 값을 T개 채운다.

#### `_discounted_reward_to_go` — t 이후 할인 보상

```python
running = 0.0
for t in reversed(range(T)):
    running = rewards[t] + self.gamma * running
    ret[t] = running
```

수식: `Q(s_t, a_t) = Σ_{t'=t}^T γ^{t'-t} r_{t'}`

뒤에서부터 순서대로 누적하면 효율적으로 계산할 수 있다. `t=T`부터 시작해서 `running = r_T`, `running = r_{T-1} + γ*r_T`, ... 식으로 역방향 재귀를 돌린다. O(T)로 계산된다.

#### `_estimate_advantage` — advantage 추정 3가지 경우

**Case 1 — baseline 없음:**
```python
advantages = q_values
```
강의에서 baseline이 없으면 reward-to-go 자체가 advantage의 역할을 한다.

**Case 2 — baseline 있음, GAE 없음:**
```python
values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)))
advantages = q_values - values
```
`A(s,a) = Q(s,a) - V(s)`. V(s)를 critic으로 추정해 Q에서 빼준다. 이렇게 하면 "평균보다 얼마나 나은가"를 측정하게 되어 분산이 줄어든다.

**Case 3 — GAE:**
```python
delta = rewards[i] + self.gamma * (1 - terminals[i]) * values[i + 1] - values[i]
advantages[i] = delta + self.gamma * self.gae_lambda * (1 - terminals[i]) * advantages[i + 1]
```

강의의 GAE 수식:
```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)          ← TD error
A_t = δ_t + (γλ)·A_{t+1}                    ← 역방향 재귀
```

`(1 - terminals[i])` 마스크가 핵심이다. 에피소드가 끝난 시점(`terminal=1`)에서는 `V(s_{t+1})`과 `A_{t+1}`을 모두 0으로 처리한다. 다음 state가 다른 에피소드에 속하기 때문에 현재 advantage 계산에 포함시키면 안 된다.

`values`에 더미값 0을 append하는 이유는 마지막 timestep(`i = batch_size - 1`)에서 `values[i+1]`을 참조할 때 index out of range를 막기 위해서다.

#### `update` — normalize advantages

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

Advantage를 정규화하면 학습이 안정적이 된다. 배치 내에서 평균 0, 표준편차 1로 만들어 gradient의 scale을 일정하게 유지한다. `1e-8`은 std가 0일 때 division by zero를 방지하는 수치 안정성 항이다.

PPO에서는 전체 배치가 아닌 **미니배치 단위**로 정규화한다는 점이 다르다. 미니배치 내의 통계로 정규화해야 미니배치 업데이트 간 scale이 일관되게 유지된다.

#### `update` — critic 업데이트 반복

```python
for _ in range(self.baseline_gradient_steps):
    critic_info = self.critic.update(obs, q_values)
```

Critic은 actor보다 더 많은 업데이트 스텝이 필요하다. Actor가 1번 업데이트될 때 critic은 `baseline_gradient_steps`번 업데이트된다. Critic이 정확해야 advantage 추정이 정확해지고, 그래야 actor가 올바른 방향으로 업데이트되기 때문이다.

#### PPO — 미니배치 업데이트 구조

```python
for _ in range(self.n_ppo_epochs):
    np.random.shuffle(inds)
    for start in range(0, n_batch, minibatch_size):
        ...
        info = self.actor.ppo_update(obs_slice, actions_slice, advantages_slice, logp_slice, self.ppo_cliprange)
```

PPO가 vanilla PG와 다른 핵심 부분이다. 같은 데이터로 `n_ppo_epochs`번 반복 학습한다. Importance sampling ratio가 policy 변화를 보정하고, clip이 과도한 변화를 막아주기 때문에 가능하다. 매 epoch마다 인덱스를 shuffle해서 미니배치 순서를 무작위로 만든다.

---

### 전체 학습 흐름 요약

```
trajectory 데이터 (lists of arrays)
   ↓
_calculate_q_vals()  →  Q값 계산 (RTG 여부에 따라)
   ↓
np.concatenate()     →  flatten (list → single array)
   ↓
_estimate_advantage()
   ├─ critic 없음: A = Q
   ├─ critic, no GAE: A = Q - V
   └─ GAE: A = Σ (γλ)^n δ_{t+n}  (역방향 재귀)
   ↓
normalize (선택)
   ↓
actor.update() or actor.ppo_update()  →  policy 업데이트
   ↓
critic.update() × baseline_gradient_steps  →  value function 업데이트
```

---