# HW4 구현 문서: Soft Actor-Critic (SAC)

## 파일 구조

- `cas4160/scripts/run_hw4.py` — 학습 루프
- `cas4160/agents/sac_agent.py` — SAC 에이전트 핵심 구현

---

## 1. 학습 루프 (`run_hw4.py`)

### 구현 내용

```python
# random_steps 이후 정책으로 액션 선택
action = agent.get_action(observation)

# 리플레이 버퍼에서 배치 샘플링
batch = replay_buffer.sample(config["batch_size"])

# 에이전트 업데이트 (batch 딕셔너리를 언패킹하여 전달)
update_info = agent.update(**batch, step=step)
```

### 설명

- `random_steps` 이전에는 탐색을 위해 무작위 액션을 사용하고, 이후부터는 학습된 정책 `agent.get_action()`으로 액션을 선택한다.
- `replay_buffer.sample()`은 `observations, actions, rewards, next_observations, dones` 키를 가진 딕셔너리를 반환하며, 이 키 이름이 `agent.update()`의 파라미터 이름과 정확히 일치하므로 `**batch`로 바로 언패킹할 수 있다.

---

## 2. SAC 에이전트 (`sac_agent.py`)

### 텐서 shape 규칙

| 변수 | Shape |
|------|-------|
| `reward`, `done` | `(batch_size,)` |
| `self.critic(obs, action)` 출력 | `(num_critics, batch_size)` |
| `entropy()` 출력 | `(batch_size,)` |
| REINFORCE의 `action` | `(num_actor_samples, batch_size, action_dim)` |
| REINFORCE의 `q_values` (critic 직후) | `(num_critics, num_actor_samples, batch_size)` |

---

### 2-1. `entropy()` — 엔트로피 근사

```python
def entropy(self, action_distribution: torch.distributions.Distribution):
    """
    Compute the (approximate) entropy of the action distribution for each batch element.
    """

    # TODO(student): Compute the entropy of the action distribution
    # HINT: use one action sample for estimating the entropy.
    # HINT: use action_distribution.log_prob to get the log probability.
    # NOTE: think about whether to use .rsample() or .sample() here
    action = action_distribution.rsample()
    entropy = -action_distribution.log_prob(action)

    assert entropy.shape == action.shape[:-1]
    return entropy
```

**구현 이유**

엔트로피의 정의는 H(π) = E[-log π(a|s)]이므로, 샘플 하나로 근사하면 `-log π(â|s)`가 된다 (과제 Eq. 6).

**`.rsample()` vs `.sample()`**

`entropy()`의 결과는 actor loss에서 `loss -= temperature * entropy`로 사용된다. 이때 backprop이 actor 파라미터(mean, std)까지 흘러야 하므로 reparametrization trick이 적용된 `.rsample()`을 사용한다. `.sample()`은 샘플 자체가 그래디언트 그래프에서 분리(detach)되어 있어 분포 파라미터로 그래디언트가 전달되지 않는다.

---

### 2-2. `q_backup_strategy()` — 타깃 Q값 전략

```python
def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
    # ...
    # TODO(student): Implement the different backup strategies.
    if self.target_critic_backup_type == "doubleq":
        next_qs = next_qs.flip(0)
    elif self.target_critic_backup_type == "min":
        next_qs = next_qs.min(dim=0).values
    else:
        # Default, we don't need to do anything.
        pass
    # ...
```

**구현 이유**

단일 critic을 그대로 타깃으로 사용하면 과대추정(overestimation) 편향이 발생한다. 이를 완화하는 두 전략을 구현했다.

**doubleq (Double-Q)**

`next_qs.flip(0)`은 critic A의 예측값을 critic B의 타깃으로, critic B의 예측값을 critic A의 타깃으로 교차시킨다. critic 인덱스 순서를 뒤집는 것만으로 교차가 구현된다. 예: `next_qs[0]`(A가 계산한 값)이 B의 타깃이 되고, `next_qs[1]`(B가 계산한 값)이 A의 타깃이 된다.

**min (Clipped Double-Q)**

두 critic 중 더 낮은(보수적인) Q값을 타깃으로 사용한다. `.min(dim=0).values`로 shape이 `(batch_size,)`로 줄어드는데, 함수 하단에서 `expand`로 `(num_critics, batch_size)`로 복원된다.

---

### 2-3. `update_critic()` — Critic 업데이트 (Bootstrapping)

```python
def update_critic(self, obs, action, reward, next_obs, done):
    (batch_size,) = reward.shape

    with torch.no_grad():
        # TODO(student): Sample from the actor
        next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)
        next_action = next_action_distribution.sample()

        # TODO(student)
        # Compute the next Q-values using `self.target_critic` for the sampled actions
        next_qs = self.target_critic(next_obs, next_action)

        next_qs = self.q_backup_strategy(next_qs)

        if self.use_entropy_bonus and self.backup_entropy:
            # TODO(student): Add entropy bonus to the target values for SAC
            # NOTE: use `self.entropy()`
            next_action_entropy = self.entropy(next_action_distribution)
            next_qs += self.temperature * next_action_entropy

        # TODO(student): Compute the target Q-value
        # HINT: implement Equation (1) in Homework 4
        target_values: torch.Tensor = reward + self.discount * (1.0 - done.float()) * next_qs

    # TODO(student): Predict Q-values using `self.critic`
    q_values = self.critic(obs, action)

    # TODO(student): Compute loss using `self.critic_loss`
    loss: torch.Tensor = self.critic_loss(q_values, target_values)

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
```

**구현 이유**

Bellman 방정식 기반의 bootstrapping으로 critic을 학습한다.

**타깃 계산을 `torch.no_grad()`로 감싸는 이유**

타깃 Q값은 고정된 라벨 역할을 해야 한다. 타깃까지 그래디언트가 흐르면 학습이 발산할 수 있기 때문에 차단한다.

**`next_action`에 `.sample()` 사용**

타깃은 `no_grad` 블록 안에 있으므로, 그래디언트 연결이 필요 없어 `.sample()`로 충분하다.

**Soft Q 타깃 (Eq. 5)**

`use_entropy_bonus and backup_entropy`가 모두 True일 때, 다음 상태의 엔트로피를 타깃 Q에 더한다:

```
y = r + γ(1-d)[Q_target(s', a') + β * H(π(·|s'))]
```

엔트로피 shape `(batch_size,)`는 `next_qs`의 `(num_critics, batch_size)`와 PyTorch가 자동 브로드캐스트한다.

**Bellman 타깃 (Eq. 1)**

```
target_values = reward + γ * (1 - done) * next_qs
```

`reward`, `done`의 shape `(batch_size,)`가 `next_qs`의 `(num_critics, batch_size)`와 브로드캐스트된다.

**MSE 손실**

`self.critic_loss = nn.MSELoss()`로, `q_values`와 `target_values` 간의 L2 오차를 최소화한다.

---

### 2-4. `actor_loss_reinforce()` — REINFORCE 정책 그래디언트

```python
def actor_loss_reinforce(self, obs: torch.Tensor):
    batch_size = obs.shape[0]

    # TODO(student): Generate an action distribution
    action_distribution: torch.distributions.Distribution = self.actor(obs)

    with torch.no_grad():
        # TODO(student): Draw self.num_actor_samples samples from the action distribution for each batch element
        # NOTE: think about whether to use .rsample() or .sample() here
        action = action_distribution.sample((self.num_actor_samples,))

        # TODO(student): Compute Q-values for the current state-action pair
        # HINT: need to add one dimension with `self.num_actor_samples` at the beginning of `obs`
        # HINT: for this, you can use either `repeat` or `expand`
        q_values = self.critic(obs.unsqueeze(0).expand(self.num_actor_samples, -1, -1), action)

        # Our best guess of the Q-values is the mean of the ensemble
        q_values = torch.mean(q_values, axis=0)

    # Do REINFORCE (without baseline)
    # TODO(student): Calculate log-probs
    log_probs = action_distribution.log_prob(action)

    # TODO(student): Compute policy gradient using log-probs and Q-values
    loss = -(log_probs * q_values).mean()

    return loss, torch.mean(self.entropy(action_distribution))
```

**구현 이유**

REINFORCE는 log π(a|s)의 그래디언트를 Q값으로 가중 평균하는 정책 그래디언트 추정기다 (과제 Eq. 9):

```
∇θ L = E[∇θ log π(a|s) * Q(s,a)]
```

따라서 loss = -E[log π * Q]로 정의하면 gradient descent가 이를 최대화한다.

**액션 샘플에 `.sample()` 사용**

Q값 계산은 `no_grad` 블록 안에서 이루어진다. REINFORCE에서는 샘플 자체가 아닌 `log_prob`을 통해 그래디언트가 흐르므로, 샘플 생성 시에는 `.sample()`로 충분하다.

**obs 차원 확장**

critic은 `(obs, action)` 쌍을 받는다. action의 shape이 `(num_actor_samples, batch_size, action_dim)`이므로, obs도 `unsqueeze(0).expand(num_actor_samples, -1, -1)`로 `(num_actor_samples, batch_size, obs_dim)` 형태로 맞춰준다. `expand`는 메모리 복사 없이 뷰를 생성한다.

**`log_prob`은 `no_grad` 밖에서 계산**

`log_probs = action_distribution.log_prob(action)`은 `no_grad` 블록 밖에 있다. `action`은 detach된 샘플이지만, `log_prob`의 그래디언트는 분포 파라미터(actor의 mean, std)로 흘러야 하므로 그래디언트 추적이 활성화된 상태에서 계산한다.

---

### 2-5. `actor_loss_reparametrize()` — Reparametrization 정책 그래디언트

```python
def actor_loss_reparametrize(self, obs: torch.Tensor):
    batch_size = obs.shape[0]

    action_distribution: torch.distributions.Distribution = self.actor(obs)

    # TODO(student): Sample actions
    # Note: Think about whether to use .rsample() or .sample() here...
    action = action_distribution.rsample()

    # TODO(student): Compute Q-values for the sampled state-action pair
    q_values = self.critic(obs, action).mean(dim=0)

    # TODO(student): Compute the actor loss using Q-values
    loss = -q_values.mean()

    return loss, torch.mean(self.entropy(action_distribution))
```

**구현 이유**

Reparametrization trick은 `a = μ(s) + σ(s)ε` (ε ~ N(0,1))으로 샘플을 표현해, Q값 자체를 actor 파라미터에 대해 직접 미분할 수 있게 한다 (과제 Eq. 10):

```
∇θ E[Q(s, a)] = E[∇θ Q(s, μ(s) + σ(s)ε)]
```

**`.rsample()` 사용 이유**

`.rsample()`은 샘플을 reparametrization 방식으로 생성하여, `action → Q(s, action) → loss`의 그래디언트가 actor 파라미터(μ, σ)까지 끊김 없이 역전파된다. `.sample()`을 쓰면 샘플이 그래프에서 분리되어 이 경로가 막힌다.

**REINFORCE와의 비교**

REINFORCE는 `log π * Q`의 그래디언트를 사용해 분산이 크고, 여러 샘플(`num_actor_samples`)이 필요하다. Reparametrize는 Q 자체를 직접 미분하므로 분산이 낮고 샘플 1개로도 충분하다.

---

### 2-6. `update()` — 전체 업데이트 루프

```python
def update(self, observations, actions, rewards, next_observations, dones, step):

    # TODO(student): Update the critic for num_critic_upates steps, and add the output stats to critic_infos
    critic_infos = []
    for _ in range(self.num_critic_updates):
        critic_infos.append(
            self.update_critic(observations, actions, rewards, next_observations, dones)
        )

    # TODO(student): Update the actor
    actor_info = self.update_actor(observations)

    # TODO(student): Perform either hard or soft target updates.
    # Relevant variables:
    #  - step
    #  - self.target_update_period (None when using soft updates)
    #  - self.soft_target_update_rate (None when using hard updates)
    # For hard target updates, you should do it every self.target_update_period step
    # For soft target updates, you should do it every step
    # HINT: use `self.update_target_critic` or `self.soft_update_target_critic`
    if self.target_update_period is not None:
        if step % self.target_update_period == 0:
            self.update_target_critic()
    else:
        self.soft_update_target_critic(self.soft_target_update_rate)
```

**구현 이유**

**Critic을 여러 번 업데이트하는 이유**

오프-정책(off-policy) 알고리즘에서는 리플레이 버퍼에 저장된 데이터를 여러 번 재사용해 critic을 집중적으로 학습시킬 수 있다. `num_critic_updates`를 높이면 데이터 효율성이 향상된다.

**타깃 네트워크 업데이트 두 가지 방식**

- **Hard update**: `target_update_period` 스텝마다 타깃 파라미터를 현재 critic으로 완전히 덮어쓴다 (`τ=1.0`). DQN에서 사용하는 방식이다.
- **Soft update (Polyak averaging)**: 매 스텝마다 `θ' ← θ' + τ(θ - θ')`로 타깃을 조금씩 현재 critic 방향으로 이동시킨다 (τ ≈ 0.005). 급격한 변화 없이 안정적으로 타깃이 갱신되어 학습이 안정된다.

`target_update_period is not None`이면 hard update, `soft_target_update_rate`가 지정된 경우 soft update를 사용한다 (두 방식 중 하나만 설정 가능).

---

## 3. 섹션 4.2 Q값 기댓값 계산

Pendulum-v1에서 항상 "아무것도 안 하는" 정책의 경우, 매 스텝 reward = -10, γ = 0.99, 에피소드가 끝나지 않는다고 가정하면:

```
Q = Σ_{t=0}^{∞} γ^t * (-10)
  = -10 / (1 - γ)
  = -10 / (1 - 0.99)
  = -10 / 0.01
  = -1000
```

따라서 학습 초기 sanity check에서 Q값이 **약 -1000** 근처에서 안정화되어야 정상이다.
