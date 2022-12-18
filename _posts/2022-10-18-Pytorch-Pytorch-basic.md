---
title: "Basic of Pytorch"
excerpt: ""

categories:
  - Pytorch
tags:
  - [Pytorch]

permalink: /Pytorch/Pytorch-basic/

toc: true
toc_sticky: true

date: 2022-10-18
last_modified_at: 2022-12-18
---


# 1. Tensor

---

## 1.1. Tensor 자료형

- tensor 자료형은 CPU tensor와 GPU tensor로 구분된다.
- default tensor : torch.FloatTensor

<img src="/assets/images/posts_img/2022-12-18-Pytorch-Pytorch-basic/1.type.png">

```python
# CPU tensor
torch.x
# GPU tensor
torch.cuda.x
```

## 1.2. torch.tensor

torch.tensor 함수는 data를 Tensor 객체로 만들어주는 함수

- data : list나 array류의 데이터
- dtype : 데이터의 타입, 선언하지 않으면 보통 data에 맞춰서 적절하게 들어간다.
- device : default는 None이나 torch.set_default_tensor_type()에 맞게 들어간다.
- requires_grad : default는 False이며, gradient 값 저장 유무를 결정한다.
- pin_memory : CPU tensor에서 가능.

**Tensor 관련 함수와 메서드**

```python
device = torch.device(’cuda’)
x = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32, device=device)

# tensor의 dtype이나 device를 바꾸는 방법
y = torch.as_tensor(x, dtype=torch.half, device='cpu')
y = x.to(device, dtype=torch.float64)

# etc
## change device
x.cpu(); x.device()

## change dtype
x.half(); x.float(); x.double()
```

**Underbar**

- 언더바가 있다면 : in_place
- 언더바가 없다면 : 새로운 tensor 리턴

거의 대부분의 메소드는 언더바 버전이 있다.

## 1.3. Copy Tensor

**tensor.new_tensor(x)**

- 근본적으로 x와 동일한 tensor를 만들어 준다.

**detach()**

- 기존 Tensor에서 gradient 전파가 안되는 tensor 형성
- storage를 공유하기 때문에 detach로 생성한 Tensor가 변경되면 원본 tensor도 똑같이 변함
    - tensor data를 임의로 조작할 때는 적합하지 않다.

**clone()**

- 기존 tensor와 내용을 복사한 텐서 생성

**clone().detach()**

- computational graph에서 더 이상 필요하지 않을 때 사용한다.
- 생성된 tensor에 어떤 연산을 진행하더라도 원래 tensor에 영향을 미치지 않는다.
- new_tensor와 동일한 작업이다.
- Reference
    
    [[Pytorch] Tensor에서 혼동되는 여러 메서드와 함수](https://subinium.github.io/pytorch-Tensor-Variable/)
    

## 1.4. tensor shape 다루기

**congiguous()**

`narrow(), view(), expand(), transpose()` 등의 함수는 새로운 Tensor를 생성하는 게 아니라 기존의 Tensor에서 메타데이터만 수정하여 우리에게 정보를 제공합니다. 즉 메모리상에서는 같은 공간을 공유합니다.

연산 과정에서 Tensor가 메모리에 올려진 순서(메모리 상의 연속성)가 중요하다면 원하는 결과가 나오지 않을 수 있고 에러가 발생한다. 함수 결과가 실제로 메모리에도 우리가 기대하는 순서로 유지하려면 `.contiguous()` 를 사용하여 에러가 발생하는 것을 방지할 수 있다.

→ 언제 쓰는지는 아직 잘 모르겠다….

**더미 차원 추가와 삭제 : squeeze() & unsqueeze()**

- squeeze() : 차원의 size가 1인 차원을 모두 없애줌
    - 특정 차원만 지우고 싶다면 squeeze(1) 처럼 index를 넣어준다.
- unsqueeze() : 차원 size가 1인 차원을 생성
    - 이것도 원하는 index에서 차원을 늘릴 수 있다.

**차원의 재구성**

**reshape() & view()**

shape 을 바꾸고 싶을 때 사용한다. `x.view(1,2,3) x.reshape(1,2,3)` 처럼 변경하고 싶은 shape를 입력으로 줘서 사용한다.

- view : 기존의 데이터와 같은 메모리 공간을 공유한다. 그래서 contigious 해야만 동작한다.
- reshape : contigious 하다면 input의 view를 반환한다. 안된다면 contiguous한 tensor로 copy하고 view를 반환한다.

view는 메모리가 기존 tensor와 동일한 메모리를 항상 공유한다. reshape는 경우에 따라 다르다.

**transpose() & permute()**

차원 간의 순서를 바꾸고 싶을 때 사용

- `transpose()` : 2개의 차원만 변경하는데 사용
- `permute()` : 모든 차원의 순서를 재배치

```python
x = torch.rand(16, 32, 3)
y = x.tranpose(0, 2) # 0<->2 차원변경
z = x.permute(2, 1, 0) # 0->2, 1->1, 2->0으로 차원변경
```

- Reference
    
    [[Pytorch] Tensor에서 혼동되는 여러 메서드와 함수](https://subinium.github.io/pytorch-Tensor-Variable/)
    
    [Pytorch란 무엇인가요? - 연산 (Pytorch 학습 2)](https://better-tomorrow.tistory.com/entry/Pytorch%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80%EC%9A%94-%EC%97%B0%EC%82%B0-Pytorch-%ED%95%99%EC%8A%B5-2)
    

## 1.5. chunk

Tensor를 자르고 분리하는데 사용하는 함수이다.

```python
import torch
x = torch.rand(4,512,512)
x_1,x_2 = torch.chunk(x,2,dim=0) # x_1.shape = (2,512,512)
```

# 2. Autograd

---

pytorch 에서 수행하는 **미분 방법**에 대해서 알아본다.

## 2.1. Autograd 사용 방법

tensor를 학습하기 위해서는 back propagation을 통하여 gradient를 구해야 한다.

tensor의 gradient를 구할 때는 다음 조건이 만족되어야 한다.

- tensor.requires_grad = True
- back propagation을 시작할 지점의 output은 **scalar** 형태
    - Q. 종종 loss를 계산할 때 .mean() .sum() 을 하던데 무슨차이?

tensor의 gradient를 구하는 방법은 backpropagation을 시작할 지점의 tensor에서 *.backward() 함수를 호출하면 된다.*

gradient 값을 확인하려면 *{tensor_name}.grad*를 통해 값을 확인 가능하다.

## 2.2. 학습 과정 in code

1. forward propagation
    - data를 model에 통과시켜 예측 값을 얻는다.
2. backward propagation
    - 계산된 loss를 가지고 backward propagation 한다.
    - 모델의 각 매개변수에 대해 gradient(변화도)를 계산하고 저장한다.
3. optimizer
    - 모델의 모든 parameter를 optimizer에 넣어준다.
4. gradient descent
    - optimizer는 parameter의 .grad에 저장된 변화도에 따라 parameter를 조정한다.
- Reference
    
    [모델 매개변수 최적화하기](https://tutorials.pytorch.kr/beginner/basics/optimization_tutorial.html)
    

## 2.3. Autograd 살펴보기

코드 실험은 Reference에 있다.

**requries_grad / grad_fn**

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
```

x

- requires_grad = True
- grad_fn = None

y

- requries_grad = True
    - requires_grad = True 옵션을 가진 tensor를 가지고 계산하면 자동으로 결과물에 True 옵션이 설정된다.
- grad_fn = <AddBackward0>
    - 이전에 어떤 계산을 했는지 알려준다.(역전파 과정의 변화도 계산에 이용)
    - 곱 계산시 : <MulBackward0>
    - 빼기 계산시 : <SubBackward0>
    - 나누기 계산시 : <DivBackward0>

**grad**

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y
z.backward()

print(z.grad) # None

print(y.grad) # None
```

tensor 의 값에 따라 계산된 gard를 나타낸다.

- grad는 tensor를 직접 정의한 변수에만 존재한다.(x)
- 계산 중간 과정인 y,z 경우 grad가 존재하지 않는다.
    - 연산에 참여한 표시로 grad_fn 옵션이 생긴다.
- Reference
    
    [PyTorch Gradient 관련 설명 (Autograd)](https://gaussian37.github.io/dl-pytorch-gradient/)
    
    [torch.autograd 에 대한 간단한 소개](https://tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html)
    

## 2.4. graph에서 parameter 제외하기

torch.autograd는 requires_grad = True 로 설정된 모든 텐서에 대한 연산들을 추적한다. 그래서 변화도가 필요하지 않은 텐서들에 대해서는 requires_grad = False 로 설정한다.

requires_grad = False 로 설정된 텐서는 gradient(변화도)를 계산하지 않기 때문에 optim.step()에도 parameter값이 변하지 않는다. 이런 매개변수를 일반적으로 frozen parameter 라고 부른다.

```python
from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# 1. requires_grad
for param in model.parameters():
    param.requires_grad = False

# 2. torch.no_grad()
with torch.no_grad():
	 z = model(x)
```

**requires_grad vs torh.no_gard()**

<img src="/assets/images/posts_img/2022-12-18-Pytorch-Pytorch-basic/2.grad.png">

model A를 업데이트 하고 model B는 업데이트하지 않기 위해서는 model B 에서 gradient를 계산은 하되  업데이트 하지 않아야 하고, model A까지 역전파는 도달해야 한다.

requires_grad와 torch.no_grad()의 차이는 위 상황에서 드러난다.

**torch.no_grad()**

- 아예 gradient 자체를 계산하지 않겠다는 의미
- model A에 전파가 되지 않는다.
- 주로 evaluation 할 때 이용한다.

**requires_grad**

- gradient를 계산하지만 업데이트는 하지 않는다.
- gradient는 여전히 흐를 수 있는 상태
- model A에 전파가 가능하다.
- pretrained model을 사용할 때 이용한다.

**{tensor}.detach()**

- z logit 뒤로 gradient 전파를 하지 않도록 독립된 tensor를 생성한다.
- 주로 fake case를 학습할 때 Discriminator의 input tensor에 이용한다.

- Reference
    
    [[PyTorch] Freeze Network: no_grad, requires_grad 차이](https://nuguziii.github.io/dev/dev-003/)
    

# 3. Network

---

## 3.1. Import

torch.nn.functional은 **함수**고 torch.nn은 **클래스**로 정의되어 있다.

개발하는 스타일에 따라서 사용하면 된다. 성능 차이는 존재하지 않는다.

- Code
    
    ```python
    import torch.nn as nn
    
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()
    
    import torch.nn.functional as F
    
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randint(5, (3,), dtype=torch.int64)
    loss = F.cross_entropy(input, target)
    loss.backward()
    ```
    

**(1) torch.nn**

- torch.nn으로 구현한 클래스의 경우에는 attribute를 활용해 state를 저장하고 활용할 수 있다.
- weight 값을 직접 설정해주지 않아도 된다.(자동으로 생성해 준다.)
- Code
    
    ```python
    # class
    torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True)
    ```
    

**(2) torch.nn.functional**

- torch.nn.functional로 구현한 함수의 경우에는 instance화 시킬 필요 없이 사용 가능하다.
- Conv layer 같은 경우 input과 weight 자체를 직접 넣어줘야 한다.
- Code
    
    ```python
    import torch.nn.Functional as F
    from torch.autograd import Variable
    filter_ = torch.ones(1,1,3,3)
    filter_ = Variable(filter_)
    out = F.Conv2d(input_,filter_)
    ```
    

- Reference
    
    [[개발팁] torch.nn 과 torch.nn.functional 어느 것을 써야 하나?](https://cvml.tistory.com/10)
    
    [[PyTorch] 3. nn & nn.functional](https://data-panic.tistory.com/9)
    

## 3.2. Layer 정의하기

**methods**

단순하게 layer를 정의하고 하나씩 쌓는 방법과(1), 가독성을 위해서 동일한 역할을 하는 module을 따로 정의하여 block으로 묶어 호출하는 방식(2,3)으로 구분된다.

1. __init__() 에 layer 객체를 정의하고 forward에서 layer객체를 쌓는 방법
2. nn.Sequential() block 객체를 만들고 block에 layer를 쌓는 방법  
3. nn.ModuleList() block 객체를 만들고 block에 layer를 쌓는 방법

def __**init__**() : 가장 기본적인 방법

torch.nn or torch.nn.functional package를 이용하여 layer들을 정의한다.

def __init__() 에서 layer를 정의하고 forward에서 layer간 연결 관계를 정의한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
```

**nn.Sequential() vs nn.ModuleList()**

nn.Sequential()은 안에 들어가는 모듈들을 연결해 준다. 자동으로 forward가 진행되며 하나의 model을 정의한다.

nn.ModuleList()는 개별적으로 모듈들이 담겨있는 **List**이다. 각 모듈 간 연결 관계들이 정의되지 않아 forward 함수 내에서 연결 관계를 정의해야 한다.  

**nn.Sequential()**

- nn.sequential 객체를 부르면 담겨있는 layer 전체가 자동으로 forward 진행된다.

```python
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
```

**nn.ModuleList()**

- forward() method가 없다.

```python
class Network(nn.Module):
    def __init__(self, n_blocks):
        super(Network, self).__init__()
        self.n_blocks = n_blocks
        block_A_list = []
        block_B_list = []
        for _ in range(n_blocks):
            block_A_list.append(Block_A())
            block_B_list.append(Block_B())
        self.block_A_list = nn.ModuleList(block_A_list)
        self.block_B_list = nn.ModuleList(block_B_list)

    def forward(self, x, k):
        for i in range(self.n_blocks):
            out = self.block_A_list[i](x)
            out = self.block_B_list[i](out, k)
        return out
```

- block_A_list에 담겨있는 n_blocks 수 만큼의 Block_A들은 서로 연결 되어있지 않다.
- forward에서 각 block들 간 input과 output이 연결되지 않아도 되도록 코드가 짜여있다. (연결 관계가 없어도 되므로) 그래서 nn.ModuleList()가 적합하다.
- Reference
    
    [신경망(Neural Networks)](https://tutorials.pytorch.kr/beginner/blitz/neural_networks_tutorial.html)
    
    [nn.ModuleList vs nn.Sequential](https://dongsarchive.tistory.com/67)
    
    [[PyTorch] nn.ModuleList 기능과 사용 이유](https://bo-10000.tistory.com/entry/nnModuleList)
    
    [7. nn.Sequential을 사용한 신경망 구현](https://dororongju.tistory.com/147)
    

# 4. Optimizer

---

딥러닝의 학습 방식은 최소의 loss를 가지는 파라미터들을 찾기 위해 미분으로 구한 기울기를 따라 이동하게 된다. 

Optimizer는 이동하는 방식을 의미하며 여러 방식 중에서 적당한 optimizer를 선택하여 학습에 이용한다.(주로 Adam을 사용한다.)

## 4.1. Optimizer 고려사항

1. Local Minima
2. Plateau
3. Zigzag

## 4.2. Optimizer 방법

**1) Gradient Descent**

- Back propagation을 통해 각 parameter마다 계산되었던 gradient 값과 learning rate를 곱한 값을 parameter에 빼준다.

$$
\theta = \theta - \alpha \frac{\delta J(\theta)}{\delta \theta}
$$

> $\alpha$  : learning rate
> 
- Code
    
    ```python
    # 구현
    class SGD:
        def __init__(self, lr=0.01):
            self.lr = lr
            
        def update(self, params, grads):
            for key in params.keys():
                params[key] -= self.lr * grads[key]
    ```
    

**2) Momentum**

- Parameter에 업데이트 할 값을 계산할 때 일종의 관성이라 할 수 있는 Momentum을 둔다.
- 직전에 계산된 gradient와 새로 계산된 gradient를 일정한 비율로 계산하는 것이다.
- gradient가 갑자기 변화해도 momentum 때문에 완만하게 parameter에 반영된다.

$$
v = \alpha v - \beta \frac{\delta L}{\delta \theta}
$$

$$
\theta = \theta + v
$$

> $\alpha$  : momentum. 상수
$\beta$ : lr. learning rate. 상수
> 
- Code
    
    ```python
    # 구현
    class Momentum:
        def __init__(self, lr=0.01, momentum=0.9):
            self.lr = lr
            self.momentum = momentum
            self.v = None
    
        def update(self, params, grads):
            if self.v is None:
                self.v = {}
                for key, val in params.items():
                    self.v[key] = np.zeros_like(val)
    
            for key in params.keys():
                self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
                params[key] += self.v[key]
    ```
    

**3) AdaGrad**

Adagrad(Adaptive Gradient)는 parameter들을 update할 때 각각의 parameter마다 step size를 다르게 설정해서 이동하는 방식.

- 지금까지 많이 변화하지 않은 parameter : step size를 크게
- 지금까지 많이 변화한 parameter : step size를 작게
(zigzag 문제를 해결하기 위해서 나온 방식인 듯.)

$$
G_t = G_{t-1} + (\nabla_{\theta}J(\theta_t))^{2}
$$

$$
\theta_{t+1} =\theta_{t} - \frac{\alpha}{\sqrt{G_t + \beta}} \bullet  \nabla_{\theta}J(\theta_t)
$$

> $\alpha$ : learning rate
$G_{t}$ : 0~t번째 step의 gradient의 누적 값
$\nabla_{\theta}J(\theta_{t})$ : t번째 step의 gradient 값
gradient 제곱값을 누적시켜 일정한 방향으로 최적화 되도록 한다.
gradient값과 반비례하게 $\theta_{t}$(parameter)를 업데이트 한다.
> 

**계속해서 값을 누적하는 형태이므로 나누어주는 수($G_{t}$)가 커져 parameter 업데이트가 느려진다.** 

**모든 weight들의 업데이트량이 비슷해지는 효과가 발생하여 학습이 느려지게 된다.**

- Code
    
    ```python
    # 구현
    class AdaGrad:
        def __init__(self, lr=0.01):
            self.lr = lr
            self.h = None
    
        def update(self, params, grads):
            if self.h is None:
                self.h = {}
                for key, val in params.items():
                    self.h[key] = np.zeros_like(val)
            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key])+1e-7)
    ```
    

**4) RMS Prop**

Adagrad의 단점을 해결하기 위한 방법. $G_{t}$부분을 **합이 아니라 지수평균**으로 대체하였다.

- $G_{t}$값이 누적되지 않고 일부만 반영되도록 수정
- gradient값에 제곱 하여 zigzag문제 해결

$$
G_{t} = \alpha G_{t-1} + (1 - \alpha)(\nabla_{\theta}J(\theta_t))^{2}
$$

$$
\theta =\theta - \frac{lr}{\sqrt{G_{t} + \beta}} \bullet  \nabla_{\theta}J(\theta_t)
$$

> $\alpha$ : decay rate
lr : learning rate
$G_{t}$ : t번째 step의 gradient 값
$\nabla_{\theta}J(\theta_{t})$ : t번째 step의 gradient 값
> 
- Code
    
    ```python
    # 구현
    class RMSprop:
        def __init__(self, lr=0.01, decay_rate = 0.99):
            self.lr = lr
            self.decay_rate = decay_rate
            self.h = None
    
        def update(self, params, grads):
            if self.h is None:
                self.h = {}
                for key, val in params.items():
                    self.h[key] = np.zeros_like(val)
    
            for key in params.keys():
                self.h[key] *= self.decay_rate
                self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
    ```
    
- Pytorch
    
    ```python
    torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    ```
    
    > weight_decay : 현재 gradient에 이전 parameter 값을 weight_decay 만큼 반영한다.($g_{t} = g_{t} + \lambda \theta_{t-1}$)
    alpha : alpha 값이 클수록 이전 gradient를 많이 이용하겠다는 의미
    eps(=$\beta$)
    momentum($\mu)$ : momentum 속성을 더해준다. 누적되는 $\frac{\alpha}{\sqrt{G_t + \beta}}$ 을 구하여 momentum 만큼 gradient에 더해준다.
    > 
    > 
    > $$
    > b_{t} = \mu b_{t-1} + \frac{g_{t}}{\sqrt{G_{t}+\beta}}
    > $$
    > 
    > $$
    > \theta_{t} = \theta_{t-1} - lr * b_{t}
    > $$
    > 
    

**5) Adam**

Adam(Adaptive Moment Estimation)은 RMSProp과 Momentum 방식을 합한 알고리즘이다. momentum값 $m_t$와 gradient값 $v_t$를 gradient에 적용한다.

- Momentum 방식과 유사하게 지금까지 계산해온 gradient의 지수 평균을 저장

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla_{\theta}J(\theta_{t})
$$

> $m_{0} = 0$
$\nabla_{\theta}J(\theta_{t})$ : t번째 step의 gradient 값
$m_{t} :$ t번째 gradient와 과거 gradient의 누적값을 지수 평균한 값
> 
- RMSProp 방식과 유사하게 기울기의 제곱 값의 지수 평균을 저장한다.

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla_{\theta}J(\theta))^{2}
$$

> $v_{0} = 0$
$\nabla_{\theta}J(\theta_{t})$ : t번째 step의 gradient 값
$v_{t} :$ t번째 gradient의 제곱과 과거 gradient 제곱의 누적값을 지수 평균한 값
> 

$$
\theta_{t} = \theta_{t-1} - lr * \frac{\hat{m_{t}}}{\sqrt{\hat{v_{t}}}+\epsilon}
$$

> $\hat{m_{t}} = \frac{m_{t}}{(1-\beta_{1}^{t})}$  , $\hat{v_{t}} = \frac{v_{t}}{(1-\beta_{2}^{t})}$
> 

**단, m과 v가 처음에 0으로 초기화되어 있기 때문에 초기 w업데이트 속도가 느리다는 단점이 있다.**

- Code
    
    ```python
    class Adam:
        def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.iter = 0
            self.m = None
            self.v = None
    
        def update(self, params, grads):
            if self.m is None:
                self.m, self.v = {}, {}
                for key, val in params.items():
                    self.m[key] = np.zeros_like(val)
                    self.v[key] = np.zeros_like(val)
    
            self.iter += 1
    				# schedule learning rate 
            lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
    
            for key in params.keys():
                #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
                #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
                self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
                self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
    
                params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
    
                #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
                #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
                #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
    ```
    
- Pytorch
    
    ```python
    torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, maximize=False)
    ```
    
    > betas($\beta_1 , \beta_2$) : 이전 까지 계산된 gradient를 얼마나 반영할 것인지 결정하는 지수 평균의 weight 계수, 값이 클수록(max=1) 이전 $m_{t-1}, v_{t-1}$를 많이 반영한다.
    weight_decay : 현재 gradient에 이전 parameter 값을 weight_decay 만큼 반영한다.($g_{t} = g_{t} + \lambda \theta_{t-1}$)
    > 
- Reference
    
    [NeuralNetwork (3) Optimazation2](https://wjddyd66.github.io/dl/NeuralNetwork-(3)-Optimazation2/)
    
    [torch.optim - PyTorch 1.11.0 documentation](https://pytorch.org/docs/stable/optim.html)
    

## 4.3. Optimizer learning rate 조절

**직접 조절**

```python
for param_group in self.optimizer.param_groups:
	param_group['lr'] = new_lr_D
```

**Scheduler 이용**

train skill 에서 설명한다.

- Reference
    
    [딥러닝 학습 향상을 위한 고려 사항들](http://www.gisdeveloper.co.kr/?p=8443)
    

# 5. Multi GPU - DDP

---

## 5.1. Operation

**1. Setup**

실무자가 여러 프로세스와 클러스터의 기기에서 계산을 쉽게 병렬화 할 수 있게 한다. 이를 위해, 각 프로세스가 다른 프로세스와 데이터를 교환할 수 있도록 메시지 교환 규약(messaging passing semantics)을 활용한다.

멀티프로레싱(`torch.multiprocessing`) 패키지와 달리, 프로세스는 다른 커뮤니케이션 백엔드(backend)를 사용할 수 있다.(’nccl’, ’gloo’ …)

여러 프로세스를 생성하는 코드

- Code
    
    ```python
    """run.py:"""
    #!/usr/bin/env python
    import os
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    
    def run(rank, size):
        """ Distributed function to be implemented later. """
        pass
    
    def init_process(rank, size, fn, backend='gloo'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)
    
    if __name__ == "__main__":
        size = 2
        processes = []
        for rank in range(size):
            p = mp.Process(target=init_process, args=(rank, size, run))
            p.start()
            processes.append(p)
    		# 다른 process가 마무리 될 때까지 기다린다.
        for p in processes:
            p.join()
    
    ```
    
    2개의 프로세스를 생성(spawn)하여 각자 다른 분산 환경을 설정하고, 프로세스 그룹(`dist.init_process_group`)을 초기화하고 최종적으로 run 함수를 실행한다.
    
    [ init process ]
    
    동일한 IP주소와 포트를 통해 마스터를 지정하고, 마스터를 통해 프로세스를 조정(coordinate)될 수 있도록 한다.
    

**2. Point-to-Point 통신**

하나의 프로세스에서 다른 프로세스로 데이터를 전송하는 것. 지점간 통신을 위해서 `send`, `recv` 함수 또는 즉시 응답하는 `isend` 와 `irecv` 함수를 사용한다.

<img src="/assets/images/posts_img/2022-12-18-Pytorch-Pytorch-basic/3.p2p.png">

[ send ]

```
torch.distributed.send(tensor, dist, group=None, tag=0)
```

- tensor : 보낼 tensor
- dist : destination rank

[ recv ]

```python
torch.distributed.recv(tensor, src, group, tag)
```

- tensor : 받은 데이터를 저장할 변수
- src : source rank. 지정하지 않고 어디든지 받아도 괜찮다면 공란
- Code
    
    ```python
    """블로킹(blocking) 점-대-점 간 통신"""
    
    def run(rank, size):
        tensor = torch.zeros(1)
        tensor2 = torch.zeros(1)
        if rank == 0:
            tensor += 2
            # Send the tensor to process 1
            dist.send(tensor=tensor, dst=1)
        else:
            # Receive tensor from process 0
            dist.recv(tensor=tensor2, src=0)
        print('Rank ', rank, ' has data ', tensor[0], tensor2[0])
        pass
    
    ```
    
    두 프로세스는 값이 0인 tensor로 시작한 후, 0번 프로세스가 tensor의 값을 1 증가시킨 후 1번 프로세스로 값을 전송하여 두 프로세스 다 1로 tensor 값이 저장된다.
    
    이 때, 프로세스 1은 수신한 데이터를 저장할 메모리를 할당해두어야 한다.
    

[ isend ]

```python
torch.distributed.isend(tensor, dst, group, tag)
```

[ irecv ]

```python
torch.distributed.irecv(tensor, src, group, tag)
```

- Code
    
    ```python
    """논-블로킹(non-blocking) 점-대-점 간 통신"""
    
    def run(rank, size):
        tensor = torch.zeros(1)
        req = None
        if rank == 0:
            tensor += 1
            # Send the tensor to process 1
            req = dist.isend(tensor=tensor, dst=1)
            print('Rank 0 started sending')
        else:
            # Receive tensor from process 0
            req = dist.irecv(tensor=tensor, src=0)
            print('Rank 1 started receiving')
        req.wait()
        print('Rank ', rank, ' has data ', tensor[0])
    
    ```
    
    즉시 응답하는 함수들을 사용할 때는 tensor를 어떻게 주고 받을지를 주의해야 한다. 데이터가 언제 다른 프로세스로 송수신되는지 모르기 때문에 `req.wait()`가 완료되기 전까지는 전송된 tensor를 수정하거나 수신된 tensor에 접근해서는 안된다.
    

**3. 집합 통신(Collective Communication)**

집합 통신은 **그룹**의 모든 프로세스에 걸친 통신 패턴을 허용한다.

- 그룹 생성 : `dist.new_group(group)` 에 순서(rank) 목록을 전달한다.
- 월드(world) : 집합 통신이 실행되는 위치
    - 예시 : 모든 프로세스에 존재하는 모든 tensor의 합을 얻기
        
        `dist.all_reduce(tensor, op, group)`
        
<img src="/assets/images/posts_img/2022-12-18-Pytorch-Pytorch-basic/4.communication.png">

[ all reduce ]

```python
torch.distributed.all_reduce(tensor, op=<ReduceOp.SUM: 0>, group, async_op)
```

- group에 있는 rank들을 모두 더하면서 rank에 값을 나눠주는 것
- 자동으로 group에 있는 ‘tensor’ 변수에 저장된 값을 더한 뒤에 대치시킨다.

[ reduce ]

```python
torch.distributed.reduce(tensor, dst, op=<ReduceOp.SUM: 0>, group, async_op)
```

- reduce는 저장되는 rank가 정해져 있어야 하기 때문에 dst parameter가 존재한다.
- dst rank의 ‘tensor’ 변수에 op 계산 값을 저장해 준다.
- Code
    
    ```python
    """ All-Reduce 예제 """
    def run(rank, size):
        """ 간단한 집합 통신 """
        group = dist.new_group([0, 1])
        tensor = torch.ones(1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        print('Rank ', rank, ' has data ', tensor[0])
    
    ```
    
    - 그룹 내의 모든 tensor들의 합이 필요하기 때문에 `dist.ReduceOp.SUM` 을 사용하였다.
    - `dist.ReduceOp.SUM` / `dist.ReduceOp.PRODUCT` / `dist.ReduceOp.MAX` / `dist.ReduceOp.Min`

# 6. Multiprocessing

```jsx
import torch.multiprocessing as mp
```

python의 multiprocessing package를 기반으로 작동한다.

## 6.1. Spawning subprocess

```python
torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')
```

- fn : function
    - fn(i, *args)
    - i : process index
- args : function에 전달할 parameter
- nprocs : number of processes to spawn
- join : perform a blocking join on all processes(block / join?)

# 7. state_dict vs parameters

---

## 7.1. parameters

> Returns an iterator over module parameters
> 

```python
model.parameters():
```

## 7.2. state_dict

> Returns containing a whole state of the module
> 

```python
model.state_dict().keys()
```

## 7.3. named_parameters

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
```

- state_dict와 named_parameter는 model안 layer의 이름과 layer에 저장되어 있는 weight를 반환하는 함수
- parameter는 그냥 weight만 반복적으로 내뱉는 듯
- model에서 원하는 layer 이름을 찾아 weight를 바꿔주고 싶을 때 state_dict, named_parameter를 사용하면 적절할 듯
- parameters는 model의 모든 parameter에 접근하고 싶을 떄 사용하면 괜찮을 듯