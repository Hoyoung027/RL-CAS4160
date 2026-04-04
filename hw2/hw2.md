# HW2 구현 노트

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