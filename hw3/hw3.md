# CAS4160 Homework 3: Q-Learning 구현 정리

## 개요

이 과제에서는 **DQN(Deep Q-Network)** 과 **Double DQN**을 구현한다.
구현은 3개의 파일에 걸쳐 있으며, 전체 학습 파이프라인은 아래 순서로 동작한다:

```
환경에서 행동 선택 (get_action)
    → 환경 스텝 실행 (env.step)
    → Replay Buffer에 저장
    → Batch 샘플링
    → Critic 학습 (update_critic)
    → Target Network 주기적 갱신 (update)
    → 평가 (sample_trajectory)
```

---

## 1. `cas4160/infrastructure/utils.py` — `sample_trajectory`

### 맥락

`sample_trajectory`는 **평가(evaluation)** 시에 사용되는 함수다.
학습 중에 주기적으로 호출되어 에이전트가 얼마나 잘하고 있는지 측정한다.
학습 루프와 분리되어 있으며, replay buffer에 데이터를 저장하지 않는다.

### 왜 중요한가

- 학습과 평가를 분리함으로써 현재 정책의 실제 성능을 측정할 수 있다.
- `terminated`와 `truncated`를 구분하는 것이 핵심이다.
  - `terminated`: 실제 에피소드 종료 (CartPole에서 막대가 쓰러짐)
  - `truncated`: 최대 스텝 도달 등 외부 이유로 강제 종료
  - 둘 다 루프를 종료시켜야 하므로 `rollout_done`은 OR 조건이다.

### 구현 코드

```python
def sample_trajectory(
    env: gym.Env, agent: DQNAgent, render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob, _ = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0

    while True:
        # TODO(student): use the most recent ob to decide what to do
        # HINT: agent.get_action()
        # 현재 관측으로 행동 선택 (epsilon은 기본값 0.02 사용 — 평가이므로 거의 greedy)
        ac = agent.get_action(ob)

        # TODO(student): take that action and get reward and next obs from the environment
        # HINT: use env.step()
        # 행동 실행
        next_ob, rew, terminated, truncated, info = env.step(ac)

        # TODO(student): rollout can end due to termination or truncation.
        # HINT: this is either 0 or 1
        # terminated OR truncated 둘 다 에피소드를 종료시킨다
        rollout_done = 1 if (terminated or truncated) else 0

        steps += 1
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(terminated)

        ob = next_ob

        if rollout_done:
            break

    episode_statistics = {"l": steps, "r": np.sum(rewards)}
    if "episode" in info:
        episode_statistics.update(info["episode"])

    env.close()

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
        "episode_statistics": episode_statistics,
    }
```

---

## 2. `cas4160/agents/dqn_agent.py` — `DQNAgent`

### 맥락

DQN의 핵심 에이전트 클래스다. 두 개의 네트워크를 유지한다:
- `self.critic`: 학습되는 **online Q-network**
- `self.target_critic`: 일정 주기로만 업데이트되는 **target Q-network**

target network를 별도로 유지하는 이유는 **Moving Target Problem** 때문이다.
정답 레이블(target value)을 계산할 때 학습 중인 네트워크를 그대로 사용하면,
정답이 매 스텝마다 바뀌어 학습이 불안정해진다.
target network를 주기적으로만 동기화함으로써 이를 방지한다.

---

### 2-1. `get_action` — epsilon-greedy 행동 선택

#### 왜 중요한가

강화학습의 핵심 딜레마인 **탐험(Exploration) vs 활용(Exploitation)**을 해결하는 전략이다.
- 탐험: 아직 안 가본 상태를 경험해서 더 좋은 전략을 발견
- 활용: 지금까지 배운 것 중 가장 좋은 행동 선택

epsilon이 높을수록 탐험, 낮을수록 활용. 학습 초반에는 epsilon을 높게 유지하고
점점 줄여나가는 것이 **epsilon scheduling**이다.

#### Q: 왜 `torch.randint`로 tensor를 만드나?

반환 직전에 `ptu.to_numpy(action).squeeze(0).item()`을 호출하는데,
이 함수는 tensor를 입력으로 받는다. 탐험/활용 두 분기 모두 동일한 타입(tensor)을
반환해야 이후 처리가 일관되게 동작한다.

#### 구현 코드

```python
def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
    """
    Used for evaluation.
    """
    observation = ptu.from_numpy(np.asarray(observation))[None]

    # TODO(student): get the action from the critic using an epsilon-greedy strategy
    if np.random.rand() < epsilon:
        # 탐험: epsilon 확률로 랜덤 행동
        action = torch.randint(self.num_actions, (1,), device=observation.device)
    else:
        # 활용: critic이 예측한 Q-value가 가장 큰 행동
        action = self.critic(observation).argmax(dim=1)

    return ptu.to_numpy(action).squeeze(0).item()
```

---

### 2-2. `update_critic` — Q-network 학습

#### 맥락

DQN 학습의 핵심 함수다. **Bellman 방정식**을 기반으로 정답 레이블을 만들고,
현재 critic의 예측과의 오차(MSE)를 줄이는 방향으로 학습한다.

```
정답(target) = r + γ * Q_target(s', a*)
예측(output) = Q_critic(s, a)
loss         = MSE(예측, 정답)
```

#### Q: `qa_values`와 `q_values`의 차이는?

- `qa_values`: critic이 출력한 **모든 행동**의 Q-value. shape: `(batch, num_actions)`
- `q_values`: 실제로 **선택했던 행동**의 Q-value만 추출. shape: `(batch,)`

`torch.gather`로 action 인덱스에 해당하는 열만 뽑아낸다.

#### Double DQN과의 차이

| | 행동 선택 네트워크 | 가치 추정 네트워크 |
|---|---|---|
| Standard DQN | `target_critic` | `target_critic` |
| Double DQN | `critic` (online) | `target_critic` |

Standard DQN은 target_critic으로 행동도 고르고 가치도 추정하기 때문에
Q-value를 과대추정(overestimation)하는 경향이 있다.
Double DQN은 행동 선택을 online critic에게 맡김으로써 이 편향을 줄인다.

#### `with torch.no_grad()`를 쓰는 이유

target value는 **정답 레이블**이므로 고정값이어야 한다.
gradient를 추적하면 불필요한 메모리와 연산이 낭비되고,
의도치 않게 target_critic까지 학습될 수 있다.

#### 구현 코드

```python
def update_critic(
    self,
    obs: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    next_obs: torch.Tensor,
    done: torch.Tensor,
) -> dict:
    """Update the DQN critic, and return stats for logging."""
    (batch_size,) = reward.shape

    # 1단계: 정답(target) 계산 — gradient 추적 불필요
    with torch.no_grad():
        # TODO(student): compute target values
        next_qa_values = self.target_critic(next_obs)  # (B, num_actions)

        if self.use_double_q:
            # Choose action with argmax of critic network
            # Double DQN: online critic으로 행동 선택
            next_action = self.critic(next_obs).argmax(dim=1, keepdim=True)
        else:
            # Choose action with argmax of target critic network
            # Standard DQN: target critic으로 행동 선택
            next_action = next_qa_values.argmax(dim=1, keepdim=True)

        next_q_values = next_qa_values.gather(1, next_action).squeeze(1)  # see torch.gather, (B,)
        # done=1이면 에피소드 종료 → 미래 보상 없음
        target_values = reward + self.discount * next_q_values * (1 - done.float())

    # TODO(student): train the critic with the target values
    # Use self.critic_loss for calculating the loss
    # 2단계: 현재 예측값 계산
    qa_values = self.critic(obs)  # (B, num_actions)
    q_values = qa_values.gather(1, action.long().unsqueeze(1)).squeeze(1)  # Compute from the data actions; see torch.gather, (B,)

    # 3단계: MSE loss 계산 및 역전파
    loss = self.critic_loss(q_values, target_values)

    self.critic_optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
        self.critic.parameters(), self.clip_grad_norm or float("inf")
    )
    self.critic_optimizer.step()
    self.lr_scheduler.step()

    return {
        "critic_loss": loss.item(),
        "q_values": q_values.mean().item(),
        "target_values": target_values.mean().item(),
        "grad_norm": grad_norm.item(),
    }
```

---

### 2-3. `update` — critic 업데이트 및 target network 갱신

#### 왜 중요한가

target network를 **매 스텝마다** 업데이트하지 않는 것이 핵심이다.
`target_update_period` 스텝마다 한 번씩만 동기화해서 정답 레이블을 안정적으로 유지한다.
너무 자주 업데이트하면 Moving Target Problem이 재발하고,
너무 드물게 하면 학습이 느려진다.

#### 구현 코드

```python
def update(
    self,
    obs: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    next_obs: torch.Tensor,
    done: torch.Tensor,
    step: int,
) -> dict:
    """
    Update the DQN agent, including both the critic and target.
    """
    # TODO(student): update the critic, and the target if needed
    # HINT: Update the target network if step % self.target_update_period is 0
    # 매 스텝마다 critic 학습
    critic_stats = self.update_critic(obs, action, reward, next_obs, done)

    # target_update_period 주기로만 target network 동기화
    if step % self.target_update_period == 0:
        self.update_target_critic()

    return critic_stats
```

---

## 3. `cas4160/scripts/run_hw3.py` — 학습 루프

### 맥락

전체 DQN 학습 파이프라인을 연결하는 스크립트다.
매 스텝마다 다음 과정을 반복한다:

1. epsilon-greedy로 행동 선택
2. 환경 스텝 실행
3. Replay Buffer에 저장
4. Batch 샘플링 후 agent 학습

### `terminated` vs `truncated` — done 플래그 처리

Replay Buffer에 저장할 때 `done` 플래그로 `terminated`만 사용한다.
`truncated`는 실제 에피소드 종료가 아니라 외부 제한(최대 스텝 초과)이므로,
TD 업데이트 관점에서는 `done=False`로 처리해야 한다.
코드에서는 `if not truncated:` 블록으로 truncated 전환은 아예 저장하지 않는 방식으로 구현되어 있다.

### 구현 코드 (핵심 부분)

```python
for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
    epsilon = exploration_schedule.value(step)

    # TODO(student): Compute action
    # HINT: use agent.get_action() with epsilon
    # 1. epsilon-greedy 행동 선택
    action = agent.get_action(observation, epsilon)

    # TODO(student): Step the environment
    # HINT: use env.step()
    # 2. 환경 스텝
    next_observation, reward, terminated, truncated, info = env.step(action)
    next_observation = np.asarray(next_observation)

    # 3. Replay Buffer 저장 (truncated는 저장하지 않음)
    if not truncated:
        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            # Atari: 마지막 프레임만 저장 (나머지는 이미 버퍼에 있음)
            replay_buffer.insert(action, reward, next_observation[-1], terminated)
        else:
            # TODO(student):
            # We're using the regular replay buffer
            # Simply insert all obs (not observation[-1])
            # 일반 환경: 전체 observation 저장
            replay_buffer.insert(observation, action, reward, next_observation, terminated)

    if terminated or truncated:
        reset_env_training()
    else:
        observation = next_observation

    if step >= config["learning_starts"]:
        # TODO(student): Sample config["batch_size"] samples from the replay buffer
        # HINT: Use replay_buffer.sample()
        # 4. 학습 (learning_starts 이후부터 시작)
        batch = replay_buffer.sample(config["batch_size"])
        batch = ptu.from_numpy(batch)

        # TODO(student): Train the agent. `batch` is a dictionary of numpy arrays.
        # HINT: agent.update
        update_info = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )
```

---

## 실험 설정 파일 목록

| 파일 | 용도 |
|---|---|
| `experiments/dqn/cartpole.yaml` | Section 4.2 기본 DQN |
| `experiments/dqn/cartpole_lr_5e-2.yaml` | Section 4.2 높은 LR 비교 |
| `experiments/dqn/bankheist.yaml` | Section 5.2 DQN (3 seeds) |
| `experiments/dqn/bankheist_ddqn.yaml` | Section 5.2 Double DQN (3 seeds) |
| `experiments/dqn/hyperparameters/cartpole_discount_0.5.yaml` | Section 6 discount 실험 |
| `experiments/dqn/hyperparameters/cartpole_discount_0.8.yaml` | Section 6 discount 실험 |
| `experiments/dqn/hyperparameters/cartpole_discount_0.99.yaml` | Section 6 discount 실험 (기본값) |
| `experiments/dqn/hyperparameters/cartpole_discount_0.999.yaml` | Section 6 discount 실험 |

## 실험 실행 명령어

```bash
cd /root/RL-CAS4160/hw3

# Section 4.2 — CartPole
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/cartpole.yaml --seed 1 -nvid 1
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/cartpole_lr_5e-2.yaml --seed 1 -nvid 1

# Section 5.2 — BankHeist DQN
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 1 -nvid 1
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 2 -nvid 1
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 3 -nvid 1

# Section 5.2 — BankHeist Double DQN
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 1 -nvid 1
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 2 -nvid 1
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 3 -nvid 1

# Section 6 — discount 하이퍼파라미터 실험
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/cartpole_discount_0.5.yaml --seed 1 -nvid 1
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/cartpole_discount_0.8.yaml --seed 1 -nvid 1
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/cartpole_discount_0.99.yaml --seed 1 -nvid 1
python cas4160/scripts/run_hw3.py -cfg experiments/dqn/hyperparameters/cartpole_discount_0.999.yaml --seed 1 -nvid 1
```
