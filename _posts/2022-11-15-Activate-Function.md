---
title: "Activate Function"
excerpt: "ReLU & Leakly ReLU & tanh & etc..."

categories:
  - DL
  - Activate Function
  
tags:
  - [DL]
  - [Activate Function]

permalink: /DL/Activate-Function/

toc: true
toc_sticky: true

date: 2022-11-15
last_modified_at: 2022-11-15
---



# 1. Kinds of Activation Funtions

---

## 1.1. Linear

<img src="/assets/images/posts_img/2022-11-15-Activate-Function/1.Linear.png">

## 1.2. ELU(Exponential Linear Unit)

<img src="/assets/images/posts_img/2022-11-15-Activate-Function/2.ELU.png">

## 1.3. ReLU(Rectified Linear Units)

<img src="/assets/images/posts_img/2022-11-15-Activate-Function/3.ReLU.png">

### Pros

- vanishing gradient problem 해결
- ReLU는 tanh, sigmoid function보다 계산 비용이 적다

### Cons

- neural network model의 hidden layer의 activation function으로 유용하다.
- ReLU에 음수 값이 들어오면 해당 노드는 학습되지 않는다.
    - 음수 값은 gradient가 0이 된다는 것을 의미하며, error / input 에 대해서 학습하지 않겠다는 의미를 가진다. 이를 dying ReLU problem이라 한다.
- ReLU은 [0,$\infin$] 범위로 output을 가지기 때문에 발산할 가능성이 있다.

## 1.4. LeakyReLU

LeakyReLU는 ReLU의 변형된 형태이다. input value가 음수일 때 0이 아닌 linear한 음수를 내놓는다. 일반적으로 음수 영역에는 graident $\alpha$ (Normally, $\alpha$ = 0.01)가 적용된 직선 graph가 존재한다.

<img src="/assets/images/posts_img/2022-11-15-Activate-Function/4.LReLU.png">

### Pros

- Leaky ReLU는 “dying ReLU” 문제를 해결하려고 고안한 function이다.

### Cons

- complex Classifiction에는 부적합하다.
    - 차라리 sigmoid나 Tanh를 쓰는게 더 낫다.

## 1.5. Sigmoid

어떤 value를 가진 input이던 간에 0과 1사이의 값으로 만들어 준다. non-linear하며 output의 범위를 제한해주고, 모든 값에 대해 서로 다른 output value를 가지도록 해주는 특징이 있다.

<img src="/assets/images/posts_img/2022-11-15-Activate-Function/5.Sigmoid.png">

### Pros

- nonlinear
- smooth gradient
- good for classifier
- output 범위가 0~1로 고정된다.

### Cons

- vanishing gradients 문제를 일으킬 수 있다.
- gradient를 죽이는 구간이 있다.
- output 범위가 zero-centered 하지 않다. 이는 gradient를 업데이트 할 때 다른 방향으로 멀리 보내게 되어 학습 속도에 영향을 미칠 수 있다.

## 1.6. Tanh

어떤 input이던 간에 -1~1범위로 만들어 준다. 

<img src="/assets/images/posts_img/2022-11-15-Activate-Function/6.Tanh.png">

### Pros

- Sigmoid 보다 tanh에서 gradient가 더 크게 계산된다.
- output이 zero-centered하다.

### Cons

- Tanh는 vanishing gradient problem이 존재한다.

## 1.7. Softmax

Softmax function calculates the probabilities distribution of the event over ‘n’ different events. In general way of saying, this function will calculate the probabilities of each target class over all possible target classes. Later the calculated probabilities will be helpful for determining the target class for the given inputs.

- Reference
    
    [Activation Functions - ML Glossary documentation](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)
    
    [갈아먹는 딥러닝 기초 [1] Activation Function(활성화 함수) 종류](https://yeomko.tistory.com/39)
    
    [Article](https://koreascience.kr/article/JAKO201835858343366.pdf)
    
    [Tips for Training Stable Generative Adversarial Networks - Machine Learning Mastery](https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/)
    
    - 여기에 generator는 relu를, discriminaor는 LReLU를 추천한다는데..? why?

# 2. In DeepLearning Network

---

## 2.1. Generator

- hidden layer : ReLU is recommaneded
- output layer : tanh or sigmoid is recommaneded

마지막 layer 에서 output 범위를 [-1, 1](tanh), [0,1](sigmoid)로 한정하기 위해서 사용한다.

## 2.2. Discriminator

- hidden layer : LeakyReLU is recomaneded
- output layer : tanh is recommaneded

> 그런데 NIPS 2016 workshop에서는 generator / discriminator 둘 다 Leaky ReLU를 사용하라고 추천함
> 

# 3. Appendix

---

[Generative adversarial networks tanh?](https://stackoverflow.com/questions/41489907/generative-adversarial-networks-tanh)

## Q1. Activation function in GAN

DCGAN에서 generator는 마지막 output layer에 tanh function을 사용하였고, 나머지 activation function은 ReLU를 사용하였다. 앞의 설정으로 모델이 color space 까지 포함하여 더 빠르게 학습하였음.

Discriminator는 LeaklyReLU가 잘 작동하는 것을 확인하였음.(특히 높은 resolution을 modeling할 때 효과적이었음)