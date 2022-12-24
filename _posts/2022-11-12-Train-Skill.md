---
title: "Train Skill"
excerpt: ""

categories:
  - DL
  - Data Augmentation
  - BN
  - Scheduler
  - MultiGPU
  - Accumulation
  
tags:
  - [DL]
  - [Data Augmentation]
  - [BN]
  - [Scheduler]
  - [MultiGPU]
  - [Accumulation]

permalink: /DL/Train-Skill/

toc: true
toc_sticky: true

date: 2022-11-12
last_modified_at: 2022-11-12
---



# 1. Data
---

## 1.1. Data Augmentation

훈련 데이터를 임의적으로 수를 늘리는 방법. 오버피팅 문제 해결과 훈련 및 테스트 데이터에 대한 정확도를 높일 수 있는 방법이다.

- Options
    
    앞에 항상 `transforms.` 가 붙는다.
    
    - `CenterCrop(size)` : 중간을 기준으로 size 만큼 자른다.
    - `ColorJitter(0,0,0,0)` : 이미지의 brightness, contrast, saturation, hue 정도를 조절한다.
    - `Grayscle()` : color image를 gray image로 바꿔준다. 채널은 1이된다.
    - `Pad(,fill,padding_mode)` : image 가장자리를 padding한다.
        - fill(default = 0)
        - padding_mode : padding 방식 설정(edge, reflect, symmetric)
    - `RandomCrop(size)` : 랜덤한 위치에서 이미지를 size만큼 crop한다.
    - `RandomHorizontalFlip(p)` : p확률로 이미지를 뒤집는다.
    - `Resize(size)` : 이미지 크기 변환
    - `GaussianBlur(kernel_size)` : 이미지에 blur 적용
    
    dtype이 Tensor일때만 사용가능한 options
    
    - `Normalize()`
    - `RandomErasing()` : 이미지의 일부를 지워준다.
    - `ToPILImage()` : tensor to PIL image

```python
import torchvision.transforms as transforms

# def transforms
transforms = transforms.Compose(
	transforms.Resize((512,512)),
	transforms.ToTensor(),
	...
)
```

- Reference
    
    [torchvision.transforms - Torchvision master documentation](https://pytorch.org/vision/0.9/transforms.html)
    

## 1.2. Data Normalization

입력되는 데이터에 대해서 공간상 분포를 정규화시켜주면 더 높은 모델의 예측 정확도를 얻을 수 있다. 전체 데이터에 대한 평균과 표준편차를 이용한다.

입력 데이터를 평균이 0, 분산이 1인 정규분포가 되도록 만드는 것을 표준화(Standardization)라 한다.

- transforms.Normalize()는 반드시 transforms.ToTensor() 뒤에 위치해야 한다.

```python
import torchvision.transforms as transforms

# def transforms
transforms = transforms.Compose(
	transforms.ToTensor()
	transforms.Normalize((평균),(표준편차))
)
```

사실 이미지의 경우 픽셀 값들이 전부 0 ~ 255 범위로 제한되어 있기 때문에 반드시 Normalize를 할 필요는 없다고도 한다.

- Options
    - transforms.ToTensor()
        - input data를 0~1 범위로 scale를 바꿔준다.
    - transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        - 0~1 범위 tensor를 -1~1 범위로 Normalize 한다.
        - 일반적으로 사용한다.
    - transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        - ImageNet 데이터셋 학습시 얻어낸 값들(ImageNet을 사용할때 필요할 듯)
        - ImageNet 데이터셋에 고품질의 데이터가 많이 존재하므로 이를 따르면 학습이 잘 될것이라 판단
- Reference
    
    [딥러닝 학습 향상을 위한 고려 사항들](http://www.gisdeveloper.co.kr/?p=8443)
    
    [Pytorch torchvision.transforms.normalize 함수](https://guru.tistory.com/72)
    
    [[Pytorch] 이미지 데이터세트에 대한 평균(mean)과 표준편차(std) 구하기](https://eehoeskrap.tistory.com/463)
    

# 2. Weight initialization

---

신경망이 깊어질 수록 각 신경망의 가중치 값들의 분포가 한쪽으로 쏠릴 수 있다. 이런 현상은 Gradient Vanishing이 발생할 수 있고, 신경망의 표현력에 제한이 발생한다.

이를 위해서 학습하기 전에 가중치를 적당하게 초기화 하는 것이 필요하다. 일반적으로 알려진 방법은 **Xavier**, **He** 등 이 존재한다.

- activation function을 ReLU, leaky_ReLU로 사용한다면 He 방법을 사용한다.
- Code
    
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.init as init
    
    class CNN(nn.Module):
        def __init__(self):
            super(CNN,self).__init__()
            self.layer = nn.Sequential(
                nn.Conv2d(1,16,3,padding=1),  # 28 x 28
                nn.ReLU(),
                nn.Conv2d(16,32,3,padding=1), # 28 x 28
                nn.ReLU(),
                nn.MaxPool2d(2,2),            # 14 x 14
                nn.Conv2d(32,64,3,padding=1), # 14 x 14
                nn.ReLU(),
                nn.MaxPool2d(2,2)             #  7 x 7
            )
            self.fc_layer = nn.Sequential(
                nn.Linear(64*7*7,100),
                nn.ReLU(),
                nn.Linear(100,10)
            )
    
            # 초기화 하는 방법
            # 모델의 모듈을 차례대로 불러옵니다.
            for m in self.modules():
                # 만약 그 모듈이 nn.Conv2d인 경우
                if isinstance(m, nn.Conv2d):
                    '''
                    # 작은 숫자로 초기화하는 방법
                    # 가중치를 평균 0, 편차 0.02로 초기화합니다.
                    # 편차를 0으로 초기화합니다.
                    m.weight.data.normal_(0.0, 0.02)
                    m.bias.data.fill_(0)
    
                    # Xavier Initialization
                    # 모듈의 가중치를 xavier normal로 초기화합니다.
                    # 편차를 0으로 초기화합니다.
                    init.xavier_normal(m.weight.data)
                    m.bias.data.fill_(0)
                    '''
    
                    # Kaming Initialization
                    # 모듈의 가중치를 kaming he normal로 초기화합니다.
                    # 편차를 0으로 초기화합니다.
                    init.kaiming_normal_(m.weight.data)
                    m.bias.data.fill_(0)
    
                # 만약 그 모듈이 nn.Linear인 경우
                elif isinstance(m, nn.Linear):
                    '''
                    # 작은 숫자로 초기화하는 방법
                    # 가중치를 평균 0, 편차 0.02로 초기화합니다.
                    # 편차를 0으로 초기화합니다.
                    m.weight.data.normal_(0.0, 0.02)
                    m.bias.data.fill_(0)
    
                    # Xavier Initialization
                    # 모듈의 가중치를 xavier normal로 초기화합니다.
                    # 편차를 0으로 초기화합니다.
                    init.xavier_normal(m.weight.data)
                    m.bias.data.fill_(0)
                    '''
    
                    # Kaming Initialization
                    # 모듈의 가중치를 kaming he normal로 초기화합니다.
                    # 편차를 0으로 초기화합니다.
                    init.kaiming_normal_(m.weight.data)
                    m.bias.data.fill_(0)
    
        def forward(self,x):
            out = self.layer(x)
            out = out.view(batch_size,-1)
            out = self.fc_layer(out)
            return out
    
    ```
    
- Reference
    
    [딥러닝 학습 향상을 위한 고려 사항들](http://www.gisdeveloper.co.kr/?p=8443)
    
    [Pytorch-학습 관련 기술들](https://wjddyd66.github.io/pytorch/Pytorch-Problem/#%EA%B0%80%EC%A4%91%EC%B9%98%EC%9D%98-%EC%B4%88%EA%B9%83%EA%B0%92)
    
    [[PyTorch] 모델 파라미터 초기화 하기 (parameter initialization)](https://jh-bk.tistory.com/10)
    
    [[ CNN ] 가중치 초기화 (Weight Initialization) - PyTorch Code](https://supermemi.tistory.com/121)
    

# 3. Layer

---

## 3.1. Batch Normalization

활성함수의 출력값을 정규화 하는 작업을 의미한다. 이는 데이터 분포가 치우치는 현상을 해결함으로써 가중치가 엉뚱한 방향으로 갱신될 문제를 해결하며. Gradient vanishing 문제를 방지한다.

<img src="/assets/images/posts_img/2022-11-12-Train-Skill/1.BN.png">


BN layer를 거치면 데이터의 분포가 평균 0, 분산 1이 되도록 정규화를 한다. model이 train mode에는 BatchNormalization을 실시하고 evaluation mode에는 BatchNormalization을 실시하지 않는다.

- Code - Pytorch
    
    ```python
    class CNN(nn.Module):
        def __init__(self):
            super(CNN,self).__init__()
            self.layer = nn.Sequential(
                nn.Conv2d(1,16,3,padding=1),  # 28 x 28
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16,32,3,padding=1), # 28 x 28
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2,2),            # 14 x 14
                nn.Conv2d(32,64,3,padding=1), # 14 x 14
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2,2)             #  7 x 7
            )
            self.fc_layer = nn.Sequential(
                nn.Linear(64*7*7,100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Linear(100,10)
            )
    
        def forward(self,x):
            out = self.layer(x)
            out = out.view(batch_size,-1)
            out = self.fc_layer(out)
            return out
    ```
    

## 3.2. Drop out

**Overfitting** 을 막기위한 방법으로 모델이 학습중일 때, 랜덤하게 뉴런을 꺼서 학습한다. 학습이 학습용 데이터로 치우치는 현상을 막아준다.

<img src="/assets/images/posts_img/2022-11-12-Train-Skill/2.DropOut.png">

pytorch에서는 model.train()으로 모델 전체에 있는 Dropout을 적용한다. model.eval()에서는 Dropout을 적용하지 않고 모든 뉴런을 이용하여 예측한다.

Dropout의 기법을 사용한다고 해서 항상 결과가 좋아지진 않는다. overfitting 하지 않는 상태에서 적용하면 오히려 학습이 잘 안되는 결과가 나온다.

- Code
    
    ```python
    class CNN(nn.Module):
        def __init__(self):
            super(CNN,self).__init__()
            self.layer = nn.Sequential(
                nn.Conv2d(1,16,3,padding=1),  # 28
                nn.ReLU(),
                nn.Dropout2d(0.2),
                nn.Conv2d(16,32,3,padding=1), # 28
                nn.ReLU(),
                nn.Dropout2d(0.2),
                nn.MaxPool2d(2,2),            # 14
                nn.Conv2d(32,64,3,padding=1), # 14
                nn.ReLU(),
                nn.Dropout2d(0.2),
                nn.MaxPool2d(2,2)             # 7
            )
            self.fc_layer = nn.Sequential(
                nn.Linear(64*7*7,100),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(100,10)
            )       
            
        def forward(self,x):
            out = self.layer(x)
            out = out.view(batch_size,-1)
            out = self.fc_layer(out)
            return out
    ```
    

# 4. Scheduler

---

## 4.1. Idea

model을 학습할 때 정해줄 수 있는 parameter 중 하나인 learning rate는 학습에 많은 영향을 미친다. 적절한 learning rate를 선택해야 모델이 정체되지 않고 loss를 낮출 수 있다.

- 너무 크게 되면 발산한다.
- 너무 작으면 학습하는데 시간이 너무 오래 걸린다.

<img src="/assets/images/posts_img/2022-11-12-Train-Skill/3.Scheduler.png">

learning rate를 학습 진행도에 따라 유동적으로 바꿔주면 빠르고 정확하게 모델을 학습시킬 수 있다. 실질적으로 Learning Rate를 크게 설정하고 점차 줄여가는 방식을 선택하기도 한다.

Pytorch 에서는 Learning rate를 학습이 진행되는 동안 점차 떨어뜨리는 방법으로 구현하였다.

`torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)`

## 4.2.Methods

### 1) Pytorch 에서 기본적으로 제공하는 scheduler

- Code
    
    ```python
    from torch.optim import lr_scheduler
    
    # 1. 먼저 model, optimizer 선언
    model = CNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # 2. scheduler 선언
    ## step size 단위로 학습률에 감마를 곱한다.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    
    ## 지정한 step 지점마다 학습률에 감마를 곱한다.
    scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[10,30,80], gamma=0.99)
    
    ## 매 epoch마다 학습률에 감마를 곱해준다.
    scheduler = lr_scheduler.ExponentialLR(optimizer,gamma= 0.99)
    
    # 위 방법들은 scheduler.step() 이 필요하지 않아보임... -> 체크 필요
    ```
    

### 2) LambdaLR

Lambda 표현식으로 작성한 함수를 통해 learning rate를 조절한다.

<img src="/assets/images/posts_img/2022-11-12-Train-Skill/4.LambdaLR.png">
<img src="/assets/images/posts_img/2022-11-12-Train-Skill/5.Lambda.png">

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# scheduler.step() 호출 될 때마다 lambda가 적용된다.
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
							 lr_lambda=lambda epoch:0.95**epoch)
```

### 3) MultiplicativeLR

Lambda 표현식으로 작성한 함수를 통해 learning rate를 조절한다. 초기 learning rate에 lambda함수에서 나온 값을 **누적곱**해서 learning rate를 계산한다.

<img src="/assets/images/posts_img/2022-11-12-Train-Skill/6.MultiLR.png">
<img src="/assets/images/posts_img/2022-11-12-Train-Skill/7.Multi.png">

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer,
               lr_lambda=lambda epoch: 0.95 ** epoch)
```

### 4) CosineAnnealingLR

learning rate가 cos 함수를 따라서 eat_min 까지 떨어졌다가 다시 초기 learning rate까지 올라온다.

<img src="/assets/images/posts_img/2022-11-12-Train-Skill/8.CosineLR.png">
<img src="/assets/images/posts_img/2022-11-12-Train-Skill/9.Cosine.png">

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
```

> T_max : 최대 iteration 횟수
> 

**이 외에도 다양한 방법이 있다.** 

**사실 언제 어떤 scheduler를 사용해야 적절한지 감이 안잡히므로 이정도로 정리한다.**


[[PyTorch] PyTorch가 제공하는 Learning rate scheduler 정리](https://sanghyu.tistory.com/113)
    

## 4.3. Print learning rate

```python
optimizer.state_dict()['param_groups'][0]['lr']
```

# 5. Multi GPU

---

학습의 성능 개선 보다는 학습의 속도 개선을 위한 방법이다.


[🔥PyTorch Multi-GPU 학습 제대로 하기](https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b)
    

# 6. Tips for stable GAN training

---

    
[https://flonelin.wordpress.com/2020/05/20/%EC%95%88%EC%A0%95%EC%A0%81%EC%9D%B8-generative-adversarial-network-%ED%8A%B8%EB%A0%88%EC%9D%B4%EB%8B%9D%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%8C%81%EB%93%A4/](https://flonelin.wordpress.com/2020/05/20/%EC%95%88%EC%A0%95%EC%A0%81%EC%9D%B8-generative-adversarial-network-%ED%8A%B8%EB%A0%88%EC%9D%B4%EB%8B%9D%EC%9D%84-%EC%9C%84%ED%95%9C-%ED%8C%81%EB%93%A4/)
    

# 7. Accumulation

---

## 7.1. Overview

Deep learning 기술이 발달하면서 model의 용량이 커지고 있다. 특히 computer vision의 high-resolution image 생성을 위해서는 GPU memory에 신경을 쓰지 않을 수 없다. GPU memory용량에 맞춰 학습을 진행하게 되면 small batch size로 설정할 것이고 결국 학습에 악영향을 미치게 된다.

이를 해결하기 위한 방법 중 하나인 ‘gradient accumulation’을 소개한다. 간단하게 이야기 하면, gradient accumulation은 small batch size를 이용하지만 gradients를 저장하고 network weight를 batches 간격으로 한번에 업데이트 하는 방법이다.

## 7.2. What is gradient accumulation

### 일반적인 학습 과정

1. 데이터를 mini-batches로 나눈다.
2. 한 batch씩 Neural network에 통과시킨다.
3. Network는 batch size 만큼 label을 예측하게 된다.
4. loss를 계산한다.
5. backward pass진행
6. update model weights

### Gradient accumulation

위 마지막 과정에서 every batch마다 weights를 업데이트 하는 대신에, gradient values를 저장하고 다음 batch로 넘어간다. 다음 batch에서 새로운 gradient를 더해주는 방식이다. Weight update는 몇 batch 진행한 다음에 진행한다.

Gradient accumulation은 larger batch size를 이용한 Network 학습을 가능하게 해준다. 만약 배치당 32개 이미지를 사용하여 학습한다고 가정한다. 그러나 내 컴퓨터는 메모리 사정으로 배치당 8개 이미지 밖에 수용하지 못한다. 이 경우 배치를 8개로 설정하고 4번 배치가 진행되면 그때 weight update를 진행 해 준다. 

## 7.3. How to make it work

- `batch idx`가 `accum_iter`로 나누어 질 때, data loader가 마지막 까지 load하였을 때 이 두가지 경우 gradient accumulation을 진행한다.
- loss를 `accum_iter` 로 나눠서 normalize를 진행한다.

```python
# batch accumulation parameter
accum_iter = 4  

# loop through enumaretad batches
for batch_idx, (inputs, labels) in enumerate(data_loader):

    # extract inputs and labels
    inputs = inputs.to(device)
    labels = labels.to(device)

    # passes and weights update
    with torch.set_grad_enabled(True):
        
        # forward pass 
        preds = model(inputs)
        loss  = criterion(preds, labels)

        # normalize loss to account for batch accumulation
        loss = loss / accum_iter 

        # backward pass
        loss.backward()

        # weights update
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()
```

## 7.4. Closing words

- always recommend using gradient accumulation when working with large architectures that consume a lof of GPU memory.