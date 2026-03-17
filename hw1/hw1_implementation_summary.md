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

### 업데이트 순서가 중요한 이유

```
zero_grad()   ← 반드시 먼저: 이전 스텝 gradient가 누적되지 않도록
backward()    ← loss에서 각 파라미터의 gradient 계산
step()        ← gradient 방향으로 파라미터 업데이트
```
