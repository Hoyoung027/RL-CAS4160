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
