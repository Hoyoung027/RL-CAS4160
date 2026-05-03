# RL-CAS4160 환경 설정 가이드 

### 1. conda 활성화

```bash
source /usr/local/bin/anaconda3/etc/profile.d/conda.sh
conda activate cas4160
```

### 2. 과제 패키지 설치

```bash
# 예시
cd ~/RL-CAS4160/hw1
pip install -e .
```

> `pip install -e .`는 서버 재시작 후에도 환경이 유지되면 생략 가능하나,
> `ModuleNotFoundError: No module named 'cas4160'` 에러가 뜨면 다시 실행.

---

## ~/.bashrc에 등록하면 편리

Vessl이 `~/.bashrc`를 유지한다면 아래를 추가해두면 새 터미널마다 자동 적용됩니다.

```bash
echo 'source /usr/local/bin/anaconda3/etc/profile.d/conda.sh' >> ~/.bashrc
echo 'conda activate cas4160' >> ~/.bashrc
source ~/.bashrc
```

## conda 환경 경로 정보

| 항목 | 경로 |
|------|------|
| anaconda3 | `/usr/local/bin/anaconda3` |
| cas4160 환경 | `/usr/local/bin/anaconda3/envs/cas4160` |
| Python | `/usr/local/bin/anaconda3/envs/cas4160/bin/python` |
| pip | `/usr/local/bin/anaconda3/envs/cas4160/bin/pip` |
