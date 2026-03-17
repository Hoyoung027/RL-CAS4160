# hw1 implementation summary

코드를 중심으로 정리하고, 의미는 짧은 주석으로만 남깁니다.

---

## 1) build_mlp

위치: [hw1/cas4160/infrastructure/pytorch_util.py](hw1/cas4160/infrastructure/pytorch_util.py)

```python
def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Activation = "tanh",
    output_activation: Activation = "identity",
) -> nn.Module:
    """
    Builds a feedforward neural network

    arguments:
        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer

        input_size: size of the input layer
        output_size: size of the output layer
        output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    # 문자열로 들어온 활성화 함수를 실제 모듈로 바꾼다.
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    # HINT 1: Take a look at the following link to see how nn.Sequential works:
    # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    # HINT 2: We are only using linear layers and activation layers.
    # HINT 3: You can simple create a list, append nn layers, and convert with nn.Sequential.
    layers = []
    in_dim = input_size  # 현재 레이어의 입력 차원

    # hidden layer를 n_layers 개수만큼 쌓는다.
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, size))
        layers.append(copy.deepcopy(activation))
        in_dim = size

    # 마지막 출력층과 출력 활성화 함수를 추가한다.
    layers.append(nn.Linear(in_dim, output_size))
    layers.append(copy.deepcopy(output_activation))

    # 순서대로 연결된 MLP를 반환한다.
    return nn.Sequential(*layers)
```

메모
- `n_layers=0`이면 hidden layer 없이 `Linear(input_size, output_size)` + `output_activation`만 생성됨.
- `copy.deepcopy`는 activation 모듈을 레이어마다 독립적으로 쓰기 위해 사용.

---

## 2) MLPPolicySL.forward

위치: [hw1/cas4160/policies/MLP_policy.py](hw1/cas4160/policies/MLP_policy.py)

```python
def forward(self, observation: torch.FloatTensor) -> Any:
    # TODO: implement the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it.
    # We are only considering continuous action cases. (we do not need to consider the case where self.discrete is True)
    # So, we would like to return a normal distirbution from which we can sample actions.
    # HINT 1: Search up documentation `torch.distributions.Distribution` object
    # And design the function to return such a distribution object.
    # HINT 2: In self.get_action and self.update, we will sample from this distribution.
    # HINT 3: Think about how to convert logstd to regular std.
    mean = self.mean_net(observation)   # 신경망으로 평균 μ 계산
    std = torch.exp(self.logstd)        # logstd → std (항상 양수 보장)
    return distributions.Normal(mean, std)  # N(μ, σ) 반환
```

### 목적

BC에서 정책은 "관측 → 행동"을 고정된 값이 아니라 **확률 분포**로 표현한다.
연속 행동 공간에서는 정규분포 N(μ, σ)를 사용한다.

- μ (평균): 신경망(`mean_net`)이 관측을 보고 계산한 "가장 그럴듯한 행동값"
- σ (표준편차): 행동의 불확실성

이 분포를 반환하면:
- `get_action`에서 `.sample()`로 실제 행동을 뽑을 수 있음
- `update`에서 `.log_prob()`으로 전문가 행동의 확률을 계산할 수 있음

### logstd vs std

표준편차 σ는 반드시 양수여야 한다. std를 직접 파라미터로 학습하면 음수가 될 수 있으므로,
대신 `logstd`를 학습한 뒤 `exp()`를 적용해 항상 양수를 보장한다.

```
std 직접 학습         → -∞ ~ +∞  ← 음수 가능, 위험
logstd 학습 후 exp()  →  0 ~ +∞  ← 항상 양수, 안전
```

---

## 3) MLPPolicySL.get_action

위치: [hw1/cas4160/policies/MLP_policy.py](hw1/cas4160/policies/MLP_policy.py)

```python
def get_action(self, obs: np.ndarray) -> np.ndarray:
    if len(obs.shape) > 1:
        observation = obs
    else:
        observation = obs[None]  # 1D 관측이면 배치 차원 추가: (ob_dim,) → (1, ob_dim)

    # TODO return the action that the policy prescribes
    # HINT 1: DO NOT forget to change the type of observation (to torch tensor).
    # Take a close look at `infrastructure/pytorch_util.py`.
    # HINT 2: We would use self.forward function to get the distribution,
    # And we will sample actions from the distribution.
    # HINT 3: Return a numpy action, not torch tensor
    observation = ptu.from_numpy(observation)   # numpy → torch tensor (GPU로 이동)
    distribution = self.forward(observation)    # forward()로 N(μ, σ) 분포 얻기
    action = distribution.sample()              # 분포에서 행동 샘플링
    return ptu.to_numpy(action)                 # torch tensor → numpy 변환 후 반환
```

### 목적

환경과 실제로 상호작용할 때 호출되는 함수.
numpy 관측을 입력받아 numpy 행동을 반환한다.

### 흐름

```
numpy obs
  → obs[None]으로 배치 차원 추가 (필요시)
  → ptu.from_numpy()로 torch tensor 변환
  → forward()로 N(μ, σ) 분포 생성
  → distribution.sample()으로 행동 샘플링
  → ptu.to_numpy()로 numpy 변환 후 반환
```

### obs[None] 이 필요한 이유

신경망은 배치 단위로 입력을 처리한다. 관측이 1D `(ob_dim,)` 이면 배치 차원을 추가해 `(1, ob_dim)` 으로 만들어야 한다.
관측이 이미 2D `(batch, ob_dim)` 이면 그대로 사용한다.

---

## 4) MLPPolicySL.update

위치: [hw1/cas4160/policies/MLP_policy.py](hw1/cas4160/policies/MLP_policy.py)

```python
def update(self, observations, actions):
    # TODO: update the policy and return the loss
    # HINT 1: DO NOT forget to call zero_grad to clear gradients from the previous update.
    # HINT 2: DO NOT forget to change the type of observations and actions, just like get_action.
    # HINT 3: DO NOT forget to step the optimizer.
    self.optimizer.zero_grad()                          # 이전 gradient 초기화
    observations = ptu.from_numpy(observations)         # numpy → torch
    actions = ptu.from_numpy(actions)                   # numpy → torch
    distribution = self.forward(observations)           # N(μ, σ) 분포 생성
    loss = -distribution.log_prob(actions).mean()       # NLL loss
    loss.backward()                                     # 역전파
    self.optimizer.step()                               # 파라미터 업데이트
    return {
        "Training Loss": ptu.to_numpy(loss),
    }
```

### 목적

전문가 행동을 모방하도록 정책 파라미터를 업데이트하는 함수.
"전문가가 한 행동이 우리 정책 분포에서 얼마나 그럴듯한가"를 최대화한다.

### 손실함수: NLL (Negative Log Likelihood)

```
loss = -log P(expert_action | observation)
```

- `distribution.log_prob(actions)`: 전문가 행동의 로그 확률 → 클수록 좋음
- 앞에 `-`를 붙여 최소화 문제로 변환 (optimizer는 최소화를 수행)
- `.mean()`: 배치 전체에 대한 평균 loss
- 따라서 loss는 "우리 정책 분포에서 전문가 행동들이 나올 로그 확률의 평균에 음수를 취한 값"을 의미


### 업데이트 순서가 중요한 이유

```
zero_grad()   ← 반드시 먼저: 이전 스텝 gradient가 누적되지 않도록
backward()    ← loss에서 각 파라미터의 gradient 계산
step()        ← gradient 방향으로 파라미터 업데이트
```

---

## 5) rollout_trajectory

위치: [hw1/cas4160/infrastructure/utils.py](hw1/cas4160/infrastructure/utils.py)

```python
def rollout_trajectory(env, policy, max_traj_length, render=False):
    # TODO: implement the following line
    ob, _ = env.reset()  # 환경 초기화, 첫 관측 획득

    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # (render 생략)
        obs.append(ob)
        ac = policy.get_action(ob)  # 현재 관측으로 행동 결정
        acs.append(ac)

        ob, rew, terminated, truncated, _ = env.step(ac)  # 환경에 행동 전달

        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # TODO end the rollout if the rollout ended
        rollout_done = (terminated or truncated) or (steps >= max_traj_length)

        terminals.append(rollout_done)
        if rollout_done:
            break

    return Traj(obs, image_obs, acs, rewards, next_obs, terminals)
```

### 목적

**카메라 1대로 영상 1편 촬영** — 에피소드 1개를 처음부터 끝까지 실행해 궤적(trajectory) 딕셔너리를 반환한다.

```
환경 리셋 → 시작
    ↓
관측 받기 → 정책으로 행동 결정 → 환경에 행동 전달 → 보상/다음관측 기록
    ↓
종료? → NO → 반복
    ↓
종료? → YES → {observation, action, reward, next_observation, terminal} 반환
```

### 종료 조건 3가지

| 조건 | 의미 |
|------|------|
| `terminated` | 환경이 자연스럽게 종료 (목표 달성, 실패 등) |
| `truncated` | 시간 제한으로 강제 종료 |
| `steps >= max_traj_length` | 설정한 최대 길이 초과 |

### 반환값

에피소드가 T 스텝 동안 진행됐다고 할 때:

```python
{
  "observation":      np.array, shape (T, ob_dim)   # 각 스텝의 관측
  "action":           np.array, shape (T, ac_dim)   # 각 스텝의 행동
  "reward":           np.array, shape (T,)           # 각 스텝의 보상
  "next_observation": np.array, shape (T, ob_dim)   # 행동 후 다음 관측
  "terminal":         np.array, shape (T,)           # 종료 여부 (마지막만 1, 나머지 0)
  "image_obs":        np.array                       # render=True일 때만 채워짐
}
```

예시 (T=3):
```
step 1: ob=[0.1, 0.2, ...] → ac=[0.5] → rew=1.0 → next_ob=[0.3, 0.4, ...] → done=0
step 2: ob=[0.3, 0.4, ...] → ac=[0.3] → rew=1.2 → next_ob=[0.5, 0.6, ...] → done=0
step 3: ob=[0.5, 0.6, ...] → ac=[0.1] → rew=0.8 → next_ob=[0.0, 0.0, ...] → done=1
```

---

## 6) rollout_trajectories

위치: [hw1/cas4160/infrastructure/utils.py](hw1/cas4160/infrastructure/utils.py)

```python
def rollout_trajectories(env, policy, min_timesteps_per_batch, max_traj_length, render=False):
    # TODO implement this function
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        traj = rollout_trajectory(env, policy, max_traj_length, render)  # 에피소드 1개 실행
        trajs.append(traj)
        timesteps_this_batch += get_trajlength(traj)  # 누적 스텝 수 업데이트

    return trajs, timesteps_this_batch
```

### 목적

**총 X분 분량이 모일 때까지 계속 촬영** — 누적 스텝 수가 `min_timesteps_per_batch` 이상이 될 때까지 에피소드를 반복 수집한다.
`bc_trainer`의 `collect_training_trajectories`에서 학습 데이터를 모을 때 사용된다.

에피소드마다 길이가 달라 "편수"가 아닌 "스텝 수"로 기준을 잡아 매 iteration마다 일정량의 데이터를 보장한다.

```
목표: 1000스텝

1편 실행 → 400스텝  누적: 400  (부족, 계속)
2편 실행 → 350스텝  누적: 750  (부족, 계속)
3편 실행 → 500스텝  누적: 1250 (충분, 종료)
```

### 반환값

```python
trajs              # list of traj dict, 길이 = 수집된 에피소드 수
timesteps_this_batch  # int, 총 누적 스텝 수 (min_timesteps_per_batch 이상)

# 위 예시라면:
trajs = [traj1, traj2, traj3]   # 3개 에피소드
timesteps_this_batch = 1250
```

---

## 7) rollout_n_trajectories

위치: [hw1/cas4160/infrastructure/utils.py](hw1/cas4160/infrastructure/utils.py)

```python
def rollout_n_trajectories(env, policy, ntraj, max_traj_length, render=False):
    # TODO implement this function
    trajs = []
    for _ in range(ntraj):
        trajs.append(rollout_trajectory(env, policy, max_traj_length, render))  # ntraj개 수집

    return trajs
```

### 목적

**정확히 N편 촬영** — `rollout_trajectory`를 정확히 `ntraj`번 호출한다.
`bc_trainer`에서 tensorboard 저장용 영상 rollout을 수집할 때 사용된다 (`MAX_NVIDEO=2`편).

```
목표: 2편

1편 실행 완료
2편 실행 완료
```

### 반환값

```python
trajs   # list of traj dict, 길이 = ntraj

# 위 예시라면:
trajs = [traj1, traj2]   # 정확히 2개 에피소드
```

