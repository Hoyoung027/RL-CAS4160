# hw1 implementation summary

## 실행 커맨드

### BC (Behavioral Cloning)

**Ant-v4**
```bash
python cas4160/scripts/run_hw1.py \
    --expert_policy_file cas4160/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
    --expert_data cas4160/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1 \
    --eval_batch_size 10000
```

**HalfCheetah-v4**
```bash
python cas4160/scripts/run_hw1.py \
    --expert_policy_file cas4160/policies/experts/HalfCheetah.pkl \
    --env_name HalfCheetah-v4 --exp_name bc_halfcheetah --n_iter 1 \
    --expert_data cas4160/expert_data/expert_data_HalfCheetah-v4.pkl \
    --video_log_freq -1 \
    --eval_batch_size 10000
```

### DAgger

**Ant-v4**
```bash
python cas4160/scripts/run_hw1.py \
    --expert_policy_file cas4160/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger \
    --expert_data cas4160/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1 \
    --eval_batch_size 10000
```

**HalfCheetah-v4**
```bash
python cas4160/scripts/run_hw1.py \
    --expert_policy_file cas4160/policies/experts/HalfCheetah.pkl \
    --env_name HalfCheetah-v4 --exp_name dagger_halfcheetah --n_iter 10 \
    --do_dagger \
    --expert_data cas4160/expert_data/expert_data_HalfCheetah-v4.pkl \
    --video_log_freq -1 \
    --eval_batch_size 10000
```

> `--eval_batch_size 10000`: ep_len(1000) 기준 약 10개 rollout 평균 → 신뢰도 있는 `Eval_AverageReturn` 확보. BC와 동일한 이유로 DAgger에도 필수.

---

## 목표

전문가(사람 또는 기존 제어기)의 행동 데이터를 모방해, 같은 상황에서 같은 행동을 하는 정책을 지도학습으로 학습한다.

## 전체 흐름

```
[사전 준비] 전문가가 환경을 직접 조작한 데이터를 파일로 저장
  expert_data.pkl  →  (obs, action) 쌍 수천 개

[BC: Behavioral Cloning]  n_iter = 1
  1. 전문가 데이터 로드 → ReplayBuffer에 저장
  2. num_agent_train_steps_per_iter 번 반복:
       버퍼에서 배치 샘플링
       → forward()로 정책 분포 N(μ, σ) 생성
       → loss = -log P(전문가 행동 | 관측)  (NLL)
       → backward() + step()  →  파라미터 업데이트
  3. 학습된 정책으로 환경 실행 → 성능 평가 (로깅)

[DAgger]  n_iter > 1  (BC의 반복적 개선 버전)
  iter 1: 위 BC와 동일
  iter 2~:
       학습된 정책으로 환경 실행 → 새 rollout 수집
       → 전문가가 그 상황에서 했을 행동으로 레이블링
       → 버퍼에 추가 (기존 + 새 데이터)
       → 합쳐진 데이터로 재학습
```

**BC의 문제점**: 학습 데이터에 없는 상황(distribution shift)이 오면 정책이 무너진다.
**DAgger의 해결**: 정책이 실제로 방문한 상황을 데이터에 추가해 반복 보완한다.

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

---

## 8) ReplayBuffer.sample_random_data

위치: [hw1/cas4160/infrastructure/replay_buffer.py](hw1/cas4160/infrastructure/replay_buffer.py)

```python
def sample_random_data(self, batch_size):
    assert (
        self.obs.shape[0]
        == self.acs.shape[0]
        == self.rews.shape[0]
        == self.next_obs.shape[0]
        == self.terminals.shape[0]
    )

    ## TODO return batch_size number of random entries
    ## from each of the 5 component arrays above.
    ## HINT 1: use np.random.choice to sample random indices.
    ## Remember not to "replace" when sampling data
    ## HINT 2: return corresponding data points from each array
    ## (i.e., not different indices from each array)
    ## You would use same indices for all arrays.
    ## HINT 3: look at the sample_recent_data function below

    indices = np.random.choice(self.obs.shape[0], batch_size, replace=False)
    return (
        self.obs[indices],
        self.acs[indices],
        self.rews[indices],
        self.next_obs[indices],
        self.terminals[indices],
    )
```

### 목적

ReplayBuffer에 쌓인 전체 transition 데이터 중 `batch_size`개를 **무작위**로 뽑아 학습에 사용한다.

### Replay Buffer가 왜 필요한가?

BC에서 한 번 수집한 rollout 데이터를 버퍼에 저장해두고 **반복해서 재사용**한다.
매 학습 스텝마다 환경을 새로 돌리지 않아도 되므로 효율적이다.

```
rollout → 데이터 수집 → ReplayBuffer에 저장
                          ↓
          매 학습 스텝마다 배치 샘플링 → 정책 업데이트
```

### 왜 random 샘플링인가?

연속된 스텝 데이터는 시간적으로 상관관계가 높다.
같은 순서로 계속 학습하면 신경망이 특정 패턴에 편향될 수 있으므로,
무작위 샘플링으로 시간적 상관관계를 깨서 더 안정적인 학습을 유도한다.

- `replace=False`: 같은 배치 안에서 동일 데이터가 중복되지 않도록 한다.

### 왜 같은 indices를 5개 배열에 모두 쓰는가?

`(obs, ac, rew, next_obs, terminal)`은 같은 time step의 데이터 쌍이다.
인덱스가 다르면 서로 관련 없는 obs와 action이 매칭되어 의미없는 학습이 된다.

```
indices = [3, 7, 1]

obs[3] ↔ ac[3] ↔ rew[3] ↔ next_obs[3] ↔ terminal[3]   ✓ 같은 transition
obs[7] ↔ ac[7] ↔ rew[7] ↔ next_obs[7] ↔ terminal[7]   ✓ 같은 transition
```

### sample_recent_data와 비교

| | `sample_random_data` | `sample_recent_data` |
|---|---|---|
| 대상 | 전체 버퍼 | 최근 데이터만 |
| 방법 | 무작위 인덱스 | `[-batch_size:]` 슬라이싱 |
| 사용 시점 | 일반 BC 학습 스텝 | DAgger에서 최근 수집 데이터 우선 활용 |

---

## 9) BCAgent

위치: [hw1/cas4160/agents/bc_agent.py](hw1/cas4160/agents/bc_agent.py)

```python
class BCAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__()
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicySL(
            self.agent_params["ac_dim"],
            self.agent_params["ob_dim"],
            self.agent_params["n_layers"],
            self.agent_params["size"],
            discrete=self.agent_params["discrete"],
            learning_rate=self.agent_params["learning_rate"],
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params["max_replay_buffer_size"])

    def train(self, ob_no, ac_na):
        # training a BC agent refers to updating its actor using
        # the given observations and corresponding action labels
        log = self.actor.update(ob_no, ac_na)  # HW1: you will modify actor.update
        return log

    def add_to_replay_buffer(self, trajs):
        self.replay_buffer.add_rollouts(trajs)

    def sample(self, batch_size):
        # HW1: you will modify this
        return self.replay_buffer.sample_random_data(batch_size)
```

### 역할

`BCAgent`는 **정책(MLPPolicySL)과 ReplayBuffer를 하나로 묶은 컨테이너**다.
`bc_trainer`는 BCAgent의 3개 메서드만 호출하고, 내부 구현(MLP, 버퍼)은 직접 접근하지 않는다.

```
bc_trainer
    │
    └── BCAgent
          ├── actor (MLPPolicySL)    ← 정책 신경망
          └── replay_buffer          ← 데이터 창고
```

### 3개 메서드 요약

| 메서드 | 하는 일 | 내부 호출 |
|---|---|---|
| `train(ob, ac)` | 정책을 1 스텝 업데이트 | `actor.update()` |
| `add_to_replay_buffer(trajs)` | rollout 데이터를 버퍼에 저장 | `replay_buffer.add_rollouts()` |
| `sample(batch_size)` | 버퍼에서 학습용 배치 샘플링 | `replay_buffer.sample_random_data()` |

### 왜 필요한가?

`bc_trainer`의 학습 루프를 깔끔하게 만들기 위한 **인터페이스 계층**이다.

```python
# bc_trainer 입장에서 학습 1 스텝:
obs, acs, _, _, _ = agent.sample(batch_size)   # 버퍼에서 꺼내고
log = agent.train(obs, acs)                    # 업데이트한다
```

trainer가 MLPPolicySL이나 ReplayBuffer의 내부 구조를 몰라도 동작한다.
나중에 정책이나 버퍼 구현을 바꿔도 trainer 코드는 수정할 필요 없다.

---

## 10) BCTrainer.collect_training_trajectories & train_agent

위치: [hw1/cas4160/infrastructure/bc_trainer.py](hw1/cas4160/infrastructure/bc_trainer.py)

```python
def collect_training_trajectories(self, itr, collect_policy, load_initial_expertdata=None):
    # TODO decide whether to load training data or use the current policy to collect more data
    # HINT1: depending on if it's the first iteration or not, decide whether to either
    # (1) If it is the first iteration and training data exists, load it using pickle.load.
    # In this case, you can directly return as follows
    # ``` return loaded_trajs, 0, None ```
    # (2) If no training data exists,
    # collect `self.params['batch_size_initial']` transitions
    # (3) If it is not the first iteration (the case of DAgger implementation),
    # collect `self.params['batch_size']` transitions
    # HINT2: use rollout_trajectories from utils
    # HINT3: you want each of these collected rollouts to be of length self.params['ep_len']

    print("\nCollecting data to be used for training...")

    if itr == 0 and load_initial_expertdata is not None:
        with open(load_initial_expertdata, "rb") as f:
            loaded_trajs = pickle.load(f)
        return loaded_trajs, 0, None  # 전문가 데이터 바로 반환

    trajs, envsteps_this_batch = utils.rollout_trajectories(
        self.env, collect_policy, self.params["batch_size"], self.params["ep_len"]
    )

    train_video_trajs = None
    if self.log_video:
        print("\nCollecting train rollouts to be used for saving videos...")
        train_video_trajs = utils.rollout_n_trajectories(
            self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True
        )

    return trajs, envsteps_this_batch, train_video_trajs


def train_agent(self):
    print("\nTraining agent using sampled data from replay buffer...")
    all_logs = []
    for train_step in range(self.params["num_agent_train_steps_per_iter"]):
        # TODO sample some data from the data buffer
        # HINT1: use the agent's sample function
        # HINT2: how much data = self.params['train_batch_size']
        ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = (
            self.agent.sample(self.params["train_batch_size"])  # 버퍼에서 배치 샘플링
        )

        # TODO use the sampled data to train an agent
        # HINT3: use the agent's train function
        # HINT4: keep the agent's training log for debugging
        train_log = self.agent.train(ob_batch, ac_batch)  # 정책 1 스텝 업데이트
        all_logs.append(train_log)
    return all_logs
```

### collect_training_trajectories 목적

매 iteration마다 학습에 사용할 trajectory 데이터를 준비한다.

```
itr == 0 AND 전문가 파일 있음
    → pickle.load()로 전문가 데이터 로드 → 바로 반환 (환경 실행 없음)

itr == 0 AND 파일 없음  /  itr > 0 (DAgger)
    → rollout_trajectories()로 현재 정책 실행해서 직접 수집

self.log_video == True
    → rollout_n_trajectories()로 영상 저장용 MAX_NVIDEO개 추가 수집
```

BC(`n_iter=1`)에서는 항상 첫 번째 경로 — 전문가 데이터를 파일에서 바로 로드한다.

### train_agent 목적


버퍼에 쌓인 데이터로 정책을 `num_agent_train_steps_per_iter`번 업데이트한다.

```
1000번 반복:
    버퍼에서 train_batch_size(=100)개 랜덤 샘플링
    → agent.train(ob, ac)  →  NLL loss 계산 → 역전파 → 파라미터 업데이트
```

`re_batch`, `next_ob_batch`, `terminal_batch`는 BC에서는 사용하지 않는다.
BC는 (관측, 행동) 쌍만으로 지도학습을 수행하기 때문이다.

### 전체 학습 루프에서의 위치

```
run_training_loop()
    │
    ├── collect_training_trajectories()  ← 데이터 준비
    ├── agent.add_to_replay_buffer()     ← 버퍼에 저장
    └── train_agent()                    ← 1000번 업데이트
```

---

## 11) BCTrainer.do_relabel_with_expert (DAgger)

위치: [hw1/cas4160/infrastructure/bc_trainer.py](hw1/cas4160/infrastructure/bc_trainer.py)

```python
def do_relabel_with_expert(self, expert_policy, trajs):
    """
    Relabels collected trajectories with an expert policy

    :param expert_policy: the policy we want to relabel the trajs with
    :param trajs: trajs to relabel
    """
    print(
        "\nRelabelling collected observations with labels from an expert policy..."
    )

    # TODO relabel collected obsevations (from our policy) with labels from an expert policy
    # HINT: query the policy (using the get_action function) with trajs[i]["observation"]
    # and replace trajs[i]["action"] with these expert labels

    for traj in trajs:
        traj["action"] = expert_policy.get_action(traj["observation"])

    return trajs
```

### 목적

BC의 핵심 문제인 **covariate shift(분포 이동)**를 해결하기 위한 DAgger의 핵심 단계.

BC는 전문가가 방문한 상태(state)에서만 학습하지만, 실제 실행 중에는 정책 실수로 전문가가 한 번도 가지 않은 상태에 진입할 수 있다. 이 함수는 우리 정책이 실제로 방문한 상태들에 대해 전문가가 했을 행동으로 다시 라벨링해 버퍼에 추가함으로써 그 간극을 메운다.

### 구현 원리

```
우리 정책이 실행한 trajectory
    obs:    [s1,  s2,  s3,  ..., sT]   ← 우리 정책이 실제 방문한 state
    action: [a1,  a2,  a3,  ..., aT]   ← 우리 정책이 선택한 action (덮어씀)

expert_policy.get_action(obs) 호출
    → 전문가가 [s1, s2, s3, ..., sT] 에서 했을 action 계산

traj["action"] 교체
    action: [a1*, a2*, a3*, ..., aT*]  ← 전문가 라벨로 교체됨
```

결과적으로 **"우리 policy의 state distribution + 전문가의 action label"** 쌍이 만들어져, 정책이 실제 방문하는 상태에서 올바른 행동을 학습할 수 있다.

### DAgger 전체 흐름에서의 위치

```
run_training_loop() (n_iter > 1, --do_dagger)
    │
    ├── iter 1: 전문가 데이터 로드 → BC와 동일하게 학습
    │
    └── iter 2~n:
          1. collect_training_trajectories()  ← 현재 정책으로 rollout 수집
          2. do_relabel_with_expert()         ← ★ 이 함수: action을 전문가 라벨로 교체
          3. agent.add_to_replay_buffer()     ← 기존 데이터 + 새 데이터 합산
          4. train_agent()                    ← 합쳐진 데이터로 재학습
          5. 평가 및 로깅
```

### BC와의 차이

| | BC | DAgger |
|---|---|---|
| 학습 데이터 source | 전문가가 사전 수집한 pkl 파일만 | 매 iter 우리 정책이 수집한 rollout |
| action 라벨 | 전문가의 원래 action | 전문가가 **우리 정책의 state**에서 했을 action |
| 데이터 누적 | 고정 | 매 iter 버퍼에 추가 (증가) |
| covariate shift | 해결 못함 | 해결 |

### get_action이 배열 전체를 받을 수 있는 이유

`traj["observation"]`은 shape `(T, ob_dim)`인 2D 배열이다.
`get_action` 내부에서 `len(obs.shape) > 1`이면 배치 차원을 추가하지 않고 그대로 처리하므로, observation 배열 전체를 한 번에 넘겨 T개의 expert action을 한 번에 얻을 수 있다.

---

## Q&A

### Q1. 이 실험은 어떤 실험인가?

BC(Behavioral Cloning)와 DAgger를 구현하고 비교하는 모방 학습 실험이다.

- **BC**: 전문가 데이터를 supervised learning으로 학습. `n_iter=1`, 전문가 데이터를 한 번만 사용.
- **DAgger**: BC의 covariate shift 문제를 해결. 매 iteration마다 현재 정책으로 rollout 수집 → 전문가에게 라벨링 요청 → 반복 학습.
- **목표**: BC가 전문가 성능의 30% 이상 달성, DAgger가 iteration마다 성능 향상.

---

### Q2. `--n_iter`와 `--num_agent_train_steps_per_iter`의 차이는?

두 개는 중첩된 루프다.

```
for itr in range(n_iter):               # ← --n_iter (바깥 루프)
    1. 데이터 수집
    2. replay buffer에 저장
    3. for step in range(num_agent_train_steps_per_iter):  # ← 안쪽 루프
           배치 샘플링 → 역전파 1회
    4. 평가 및 로깅
```

- `--n_iter`: "데이터 수집 → 학습 → 평가" 사이클을 몇 번 반복할지. BC는 반드시 1.
- `--num_agent_train_steps_per_iter`: 1 iteration 안에서 gradient update를 몇 번 할지. 데이터는 고정된 채 가중치만 업데이트.

---

### Q3. 1 step이란?

```
replay buffer에서 미니배치 100개 샘플링
    → 정책이 action 예측
    → 예측 action vs 전문가 action → MSE/NLL loss 계산
    → loss.backward() → 가중치 업데이트
```

1 step = 미니배치 1번 뽑아서 역전파 1회. 1000 step = 이걸 1000번 반복.

---

### Q4. TensorBoard에서 어떤 지표를 확인해야 하나?

| 지표 | 의미 |
|------|------|
| `Eval_AverageReturn` | 학습된 정책의 평균 점수 |
| `Eval_StdReturn` | 그 점수의 표준편차 |
| `Initial_DataCollection_AverageReturn` | 전문가 점수 (기준값) |

**BC 통과 기준**: `Eval_AverageReturn` ≥ `Initial_DataCollection_AverageReturn` × 0.3

DAgger에서는 iteration이 늘수록 `Eval_AverageReturn`이 올라가는 학습 곡선을 확인.

---

### Q5. rollout이란?

**1 rollout = 정책이 환경에서 처음부터 끝까지 한 번 플레이한 것 (게임 1판)**

```
reset() → 초기 상태
  step 1: 관측 → 정책이 action 출력 → 환경 진행
  step 2: 관측 → 정책이 action 출력 → 환경 진행
  ...
  step 1000: 종료
```

여러 rollout의 return을 평균 내면 `Eval_AverageReturn`, 표준편차가 `Eval_StdReturn`.

---

### Q6. BC 학습에서 rollout이 어떻게 적용되나?

| 단계 | rollout 주체 | 목적 |
|------|-------------|------|
| 학습 데이터 수집 | 전문가 (pkl 파일, 미리 저장된 것) | BC 학습용 |
| 평가 | BC 정책 (학습된 것) | 성능 측정용 |

BC는 학습 시 환경을 직접 돌리지 않는다. 전문가가 미리 뛴 데이터로만 학습하고, 평가할 때만 직접 rollout 실행.

---

### Q7. 과제 명세에서 왜 "multiple rollout"을 강조하는가?

1개 rollout만 하면 운에 따라 점수가 크게 달라져 신뢰도가 낮다. 여러 번 돌려야 통계적으로 의미있는 평균/표준편차가 나온다.

`eval_batch_size=10000`, `ep_len=1000` → **약 10개 rollout** → 신뢰도 있는 `Eval_AverageReturn ± Eval_StdReturn`.

기본값 `eval_batch_size=1000`은 ep_len과 같아서 rollout이 1개밖에 안 되므로 과제 요구사항 미충족.

---

### Q8. `--eval_batch_size`와 `--ep_len`을 따로 지정해야 하나?

`--ep_len`은 건드릴 필요 없다. Ant-v4는 자동으로 1000으로 설정된다.
`--eval_batch_size`만 늘려주면 된다.

```bash
--eval_batch_size 10000   # ep_len(1000) × 10 → 약 10개 rollout
```