# HW5: Offline RL — CQL Implementation

## 실험 실행 명령어 (nohup 병렬)

```bash
cd /root/RL-CAS4160/hw5

nohup python cas4160/scripts/run_cql.py --env_name PointmassMedium-v0 \
    --exp_name cql_alpha_0.0_expert --dataset_name expert --cql_alpha=0.0 \
    > data/expert_alpha0.0.log 2>&1 &

nohup python cas4160/scripts/run_cql.py --env_name PointmassMedium-v0 \
    --exp_name cql_alpha_0.1_expert --dataset_name expert --cql_alpha=0.1 \
    > data/expert_alpha0.1.log 2>&1 &

nohup python cas4160/scripts/run_cql.py --env_name PointmassMedium-v0 \
    --exp_name cql_alpha_0.0_random --dataset_name random --cql_alpha=0.0 \
    > data/random_alpha0.0.log 2>&1 &

nohup python cas4160/scripts/run_cql.py --env_name PointmassMedium-v0 \
    --exp_name cql_alpha_0.1_random --dataset_name random --cql_alpha=0.1 \
    > data/random_alpha0.1.log 2>&1 &
```

- `nohup ... &` : 터미널을 닫아도 백그라운드에서 계속 실행됨
- `> data/xxx.log 2>&1` : stdout/stderr를 data/ 폴더에 저장 (run_cql.py가 자동 생성하는 폴더)
- `jobs` 또는 `ps aux | grep run_cql` 로 실행 중인 프로세스 확인
- `tail -f data/expert_alpha0.1.log` 로 실시간 로그 확인

---

## 구현 내용: `CQLCritic.update()`

파일 위치: `cas4160/critics/cql_critic.py`

### 전체 코드

```python
def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
    ob_no = ptu.from_numpy(ob_no)
    ac_na = ptu.from_numpy(ac_na).to(torch.long)
    next_ob_no = ptu.from_numpy(next_ob_no)
    reward_n = ptu.from_numpy(reward_n)
    terminal_n = ptu.from_numpy(terminal_n)

    # [구현] CQL Loss 계산 (Equation 1)
    dqn_loss, qa_t_values, q_t_values = self.dqn_loss(ob_no, ac_na, next_ob_no, reward_n, terminal_n)
    q_t_logsumexp = torch.logsumexp(qa_t_values, dim=1)
    cql_loss = (q_t_logsumexp - q_t_values).mean()
    loss = dqn_loss + self.cql_alpha * cql_loss

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    info = {'Training Loss': ptu.to_numpy(loss)}
    info['CQL Loss'] = ptu.to_numpy(cql_loss)
    info['Data q-values'] = ptu.to_numpy(q_t_values).mean()
    info['OOD q-values'] = ptu.to_numpy(q_t_logsumexp).mean()

    self.learning_rate_scheduler.step()
    return info
```

### 구현 설명

CQL의 목적 함수 (Equation 1):

```
L_Q(θ) = [TD오차]  +  α × [CQL regularizer]

         1/N Σ (Q(s,a) - (r + γ max Q(s',a')))²
       + α × 1/N Σ (log Σ_a exp(Q(s,a))  -  Q(s, a_i))
```

#### 1. `dqn_loss, qa_t_values, q_t_values = self.dqn_loss(...)`

`dqn_loss()` 메서드가 세 가지를 반환한다:
- `dqn_loss` : TD오차 (MSE). 수식의 첫 번째 항
- `qa_t_values` : `Q(s_i, a)` for **모든** 행동 → shape `[batch, ac_dim]`
- `q_t_values` : `Q(s_i, a_i)` 실제 **선택된** 행동만 → shape `[batch]`

#### 2. `q_t_logsumexp = torch.logsumexp(qa_t_values, dim=1)`

수식의 `log Σ_a exp(Q(s_i, a))` 항.  
`dim=1`을 지정해 배치 내 각 상태(행)마다 모든 액션(열)에 대해 logsumexp를 계산한다.  
결과 shape: `[batch]`

#### 3. `cql_loss = (q_t_logsumexp - q_t_values).mean()`

수식의 CQL regularizer.  
- `q_t_logsumexp` : OOD 행동을 포함한 모든 행동의 Q값 (log-sum-exp로 softmax처럼 집계)
- `q_t_values` : 데이터셋에 실제 있는 행동의 Q값  
- 차이의 평균을 minimize → 데이터에 없는 행동의 Q값이 과대추정되지 않도록 억제

#### 4. `loss = dqn_loss + self.cql_alpha * cql_loss`

최종 loss. `cql_alpha=0.0`이면 CQL regularizer 항이 사라져 일반 DQN과 동일해진다.
