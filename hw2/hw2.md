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