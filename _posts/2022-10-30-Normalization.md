---
title: "Normalization"
excerpt: ""

categories:
  - DL
  - Layer
  - Normalization
  
tags:
  - [DL]
  - [Layer]
  - [Normalization]

permalink: /DL/Normalization/

toc: true
toc_sticky: true

date: 2022-10-30
last_modified_at: 2022-10-30
---


# 1. In this article…
---

- review and understand the most common normalization methods.
- Different methods have been introduced for different tasks and architectures.
- We will attempt to associate the tasks with the methods although some approaches are quite general.

# 2. Introduction
---

## 2.1. Importance range of features values

why we want normalization inside any model.

Imagine what will happen if the first input features are lying in different ranges.

어떤 input feature의 값 범위가 [0, 1] 그리고 다른 input feature의 값 범위가 [0,10000] 이라면? 모델의 weight가 작고 근사한 범위의 값을 가지기 때문에 [0,1] 경우를 무시할 수 있다. 그래서 Normalization이 상당히 중요하다.

## 2.2. In model architecture

이런 현상은 모델 안에서도 발생할 수 있다.

💡 **If we think out of the box, any intermediate layer is conceptually the same as the input layer: it accepts features and transforms them.**

deep learning model은 여러 layer로 이루어져 있고, layer은 features를 input으로 받기 때문이다. features 역시 어떤 범위의 값을 가지기 때문에 이를 조절해 줄 필요가 있다.

<img src="/assets/images/posts_img/2022-10-30-Normalization/1.Norm-graph.png">

# 3. Kind of Normalization
---

## 3.1. Notations

- Terms
    - N : batch size
    - H : height
    - W : width
    - C : channels
    - $\mu()$ : mean
    - $\sigma()$ : standard deviation
    - y : segmentation mask
    - m : just mask

<img src="/assets/images/posts_img/2022-10-30-Normalization/2.axis.png">

$$
x,y,m \in R^{N*C*H*W}
$$

- 4D activation map을 3D shape로 시각화하기 위해 H, W 두 차원을 합친다.

<img src="/assets/images/posts_img/2022-10-30-Normalization/3.Norm-axis.png">

## 3.2. Batch Normalization (2015)

💡 **Batch Normalization (BN) normalizes the mean and standard deviation for each individual feature channel/map.**

BN은 batch로 들어온 feature map의 각 channel간 평균과 표준편차를 normalize해준다.

이미지 특성으로서 평균과 표준편차는 first-order statistics라 할 수 있다. 그래서 이것들은 image style처럼 image가 가지는 **global characteristics**와 관련이 있다.

이 BN 방법은 feature map channel 간에 characteristics를 공유하도록 하기 위한 전략으로 많이 선택된다.(구체적으로 feature map channel 간 공통된 특성을 잘 파악하기 위해서 사용한다.) 이러한 이유로 BN이 downstream task에 많이 사용된다.(i.e. image classification)

> Feature map의 channel에는 conv weight와 계산한 결과가 있다. Batch size만큼 들어온 input들의 각 feature map에서 동일한 위치의 channel들은 같은 conv weight와 계산한 결과들이 위치할 것이다. BN은 channel별로 mean과 standard deviation을 계산한다고 하였으므로, batch size image의 feature map channel간에 분포를 공유할 수 있을 것이다.
> 

수학적 관점으로 보면, **Normalization은 요소들을 특정 범위 내로 한정할 수 있으므로, feature map이 BN을 거치게 되면 channel별로 같은 분포를 가지게 된다.** 

<img src="/assets/images/posts_img/2022-10-30-Normalization/4.BN.png">

$$
BN(x) = \gamma(\frac{x - \mu(x)}{\sigma(x)}) + \beta
$$

이렇게 feature values를 Gaussian-like space로 만들어서 모델을 잘 학습시킬 수 있다.

위 BN의 수학 식에서 $\gamma$와 $\beta$는 **trainable parameters** 로써 linear/affine transformation 이 되도록 한다. 이 값들은 channel별로 다르며, 개수는 channel의 수와 동일하다.

### In Pytorch

```python
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

- 여기서 affine option이 $\gamma$와 $\beta$를 사용할지 말지 선택하는 것이다.

### Advantages of BN

- BN은 deep neural network의 학습을 가속화 시킨다.
- 매 mini-batch마다 다른 값을 계산하고 regularization 한다. Regularization은 복잡한 deep neural network의 학습을 완화시켜주는 역할을 한다.
- 매 mini-batch마다 다른 분포를 가지고 있다. 이를 **mini-distributions Internal Covariate Shift**라 한다. BN은 이 현상을 해결해 준다.
- BN은 network를 통과하는 gradient flow에도 이점을 준다. gradient의 parameters의 initial values에 대한 의존도를 줄여준다. 이는 높은 learning rate를 설정 가능하도록 한다.
- nonlinearities network

### Disadvantages of BN

- 매우 작은 batch size로 학습할 때 사용하기엔 적합하지 않다.(i.e. video prediction, segmentation and 3D medical image processing) 이는 model의 error을 일으킨다.
- Problems when batch size is varying. Example showcases are training VS inference, pretraining VS fine tuning, backbone architecture VS head.

### In test

학습때 얻은 Batch size 만큼의 평균과 분산을 이용한다.

→ feature map의 값에 따라 평균과 분산을 계산하는 것이 아니라 지정된 평균과 분산을 이용한다.

> 각 batch 별 feature map의 평균과 분산값이 무시될 수 있어보인다.
> 


## 3.3. Synchronized Batch Normalization (2018)

Training Scale이 커질수록 BN은 필수적이다. BN이 나온 이후 업그레이드 버전으로 Synchronized BN(**Synch BN**) 이 나오게 되었다. **Synchronized**의 의미는 각 GPU에서 별도로 mean과 variance를 업데이트하지 않는다는 의미이다.(같은 값으로 통일해서 업데이트 하는듯)

💡 **Instead, in multi-worker setups, Synch BN indicates that the mean and standard-deviaton are communicated across workers (GPUs, TPUs etc).**

<img src="/assets/images/posts_img/2022-10-30-Normalization/5.AdaIN.png">

## 3.4. Layer Normalization (2016)

BN은 batch와 spatial dims 단위로 계산하는 방법이었다. 반대로 **Layer Normalization(LN)**은 모든 channels과 spatial dims에 대해서 Normalization 한다. 그래서 LN은 batch와 독립적인 관계를 가진다. 이 layer는 handle vectors로 처음에 정해진다.(mostly the RNN outputs.)

Transformers 논문에서 나오기 전까지는 주목받지 못한 Norm이다.

> LN은 batch와 독립적인 Normalization 이므로 적은 batch size에도 사용할 수 있다. 사실상 중간 layer의 output feature map을 통째로 Norm하는 것이기 때문에 Layer Normalization이라 이름을 붙인 것 같다.

computation of mean and std in Synch BN

$$
\sigma^{2} = \frac{\sum^{N}_{i=1}(x_{i} - \mu)^{2}}{N} = \frac{\sum^{N}_{i=1}x^{2}_{i}}{N} - \frac{(\sum^{N}_{i=1})^{2}}{N^{2}}
$$

## 3.5. Instance Normalization : The Missing Ingredient for Fast Stylication (2016)

💡 **Instance Normalization (IN) is computed “only across the features’ spatial dimensions”. So it is independent for each channel and sample.**

BN과 IN을 비교하자면 BN에서 *N dimension*을 제외하고 Normalization한 것이 IN이다. 오로지 spatial dimension 차원에 대해서 normalization 하는 방식이다.  IN은 each individual sample의 style 정보를 mean과 standard deviation을 뽑아낼 수 있으며, 이를 이용하면 denormalization으로 다른 이미지의 스타일을 바꿔주는 것도 가능하다.(modeled by $\gamma,\space \beta$)

이로써 스타일 정보를 뽑기도 하고 전달할 수 있게 되었다. style 정보를 받은 모델은 style information을 학습하는데 주의를 기울이지 않아도 되었고 content manipulation, local detail 같이 다른 부분을 학습하는데 focusing이 가능해져 학습을 수월하게 한다.

$$
IN(x) = \gamma\frac{x-\mu(x)}{\sigma(x)} + \beta
$$

## 3.6. Adaptive Instance Normalization (2017)

Normalization과 style transfer는 관련도가 높다. IN에서 $\gamma, \space\beta$는 스타일 관련 정보를 가지고 있기 때문에 image $y$ 에 $\gamma, \space\beta$ 로 denormalization을 해준다면 image y는 image x의 style을 가지게 된다.

💡 **Adaptive Instance Normalization (AdaIN) receives an input image x*x* (content) and a style input y*y*, and simply aligns the channel-wise mean and variance of x to match those of y. Mathematically:**


$$
AdaIN(x,y) = \sigma(y)\frac{x - \mu(x)}{\sigma(x)} + \mu(y)
$$

> feature map의 channel별 mean과 standard deviation은 input image의 style과 관련된 정보를 가지고 있다.
input image의 structure 정보는 어떻게 찾을 수 있을지??

## 3.7. Group Normalization (2018)

GN은 deeplearning 이전 object detection task를 위해 이용된 HOG feature 방식을 따른다.

**HOG feature 방식 순서**
1. image안에 여러 patch 영역을 정한다.
2. 각 patch 영역별로 histogram을 계산하고 Normalization을 진행한다.
3. concatenate 하여 최종 feature map을 얻는다.

GN역시 channel을 N개의 group으로 나누고 group 별로 Normalization 하여 feature map을 얻는다.

💡 **Group normalization (GN) “divides” the channels into groups and computes the first-order statistics within each group.**

GN은 **큰 batch size로** 학습하는 BN보다 안정되게 학습되는 특징이 있다. 

💡 **For groups=number of channels we get instance normalization, while for`groups=1 the method is reduced to layer normalization.**

$$
\mu_{i} = \frac{1}{m}\sum_{k\in S_{i}}, \space \sigma_{i} = \sqrt{\frac{1}{m}\sum_{k\in S_{i}}(x_{k} - \mu{i})^{2}+\epsilon}
$$

$$
S_{i} = \left\{ {k|k_{N}}, \left [ \frac{kC}{\frac{C}{G}}\right ] = \left [ \frac{iC}{\frac{C}{G}} \right ] \right \}
$$

- G : the number of groups(hyper parameter)
- C/G : the number of channels per group
- GN은 group수 만큼 mean과 standard deviation을 계산하게 된다.

> GN이 object detection task를 위해서 고안된 Normalization 종류라 생각된다.


## 3.8. Spectral Normalization for Generative Adversarial Networks (2018)

GAN 학습에서 불안정성이 계속해서 언급되어 왔음. 이에 discriminator의 훈련을 안정화시킬 수 있는 spectral normalization을 새롭게 제안하였음. 

- 계산량을 줄여줌
- 쉽게 적용 가능
- 학습을 안정시킴

이전에도 weight를 normalization하는 방법이 있었다. weight normalization, weight clipping(WGAN), gradient penalty(WGAN-GP)

# 4. Appendix
---

## 4.1. GN

### GN vs else Normalizaiton

<img src="/assets/images/posts_img/2022-10-30-Normalization/6.GN.png">
<img src="/assets/images/posts_img/2022-10-30-Normalization/7.compare.png">

- BN사용 시 Batch size는 16이상은 의미가 없어 보임
- GN은 작은 Batch size에도 적은 error을 보임
    - 안정적인 학습 가능
- 동일한 Batch size로 각각 학습하면 GN이 더 학습 잘됨

### Amount of groups

<img src="/assets/images/posts_img/2022-10-30-Normalization/8.val.png">

- 실험 상 8, 32개가 가장 적합.
- batch size를 32개 이상 해도 될 듯?

## 4.2. Norm layer in model architecture

## Q1

[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/782](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/782)

### Q1.1 Pix2Pix에서 first down-sampling block에 IN을 적용하지 않음. 왜?

IN의 특성은 input의 style이나 color 정보를 삭제하는 경향을 지닌다. 저자는 이 style 정보를 조금 더 보존하고 싶어 첫 block에 IN을 적용하지 않았다고 한다.

### Q1.2. 그렇다면 BN을 first block에 적용하면 안되나?

BN은 individual image보다 전체 dataset에 대해 통계를 계산해 준다. 그리고 BN으로 각 이미지의 style 정보를 보존할 수 있을 듯.

> 개인적인 생각으로 individual image의 style을 encoder를 통해 뽑아내고 싶다면 IN이 적합해 보인다.
IN을 통해서 norm 되는 정보는 color인지 style인지 둘 다 인지 의문

## Q2

[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/981](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/981)

### Q2.1. 왜 CycleGAN 구조에서 first block에서 IN을 적용했는가?

아까 Q1.1. 과 반대되는 경우다. CycleGAN은 resnet-based architecture를 따른 것이다. architecture를 보면 first block에 큰 Conv layer(7x7)가 norm layer 전에 존재한다. 이러면 color information이 충분히 encode 된 것이라 보기 때문에 IN을 넣은 것이다.

### Q2.2. use_bias = norm_layer == nn.InstanceNorm2d 의 의미는?

flag `affine` 에 달려있는 옵션이다. IN layer의 option이 `affine=True` 라면 Conv layer에서 `bias=False` 를 설정해 줘야 한다.

어차핀 Norm의 affine에 의해서 bias는 의미가 없어져 버리기 때문이다.

**Reference**

[Batch Normalization(BN)](https://kjhov195.github.io/2020-01-09-batch_normalization/)
    
[PyTorch에서 다중 GPU 동기화 배치 정규화 구현 - wenyanet](https://www.wenyanet.com/opensource/ko/603fe7a794216c52da4597e4.html)

[In-layer normalization techniques for training very deep neural networks AI Summer](https://theaisummer.com/normalization/)

[instance vs BN](https://www.baeldung.com/cs/instance-vs-batch-normalization)

[논문 Summary SNGAN (2018 ICLR) "Spectral Normalization for Generative Adversarial Networks"](https://aigong.tistory.com/371#%EB%85%BC%EB%AC%B8_%EB%A7%81%ED%81%AC)

[Group Normalization](https://youtu.be/m3TN9FFmqsI)

[Conv2d - PyTorch 1.12 documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

[InstanceNorm2d - PyTorch 1.12 documentation](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html)
