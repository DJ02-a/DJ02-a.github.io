---
title: "Discriminator"
excerpt: ""

categories:
  - DL
  
tags:
  - [DL]

permalink: /DL/Discriminator/

toc: true
toc_sticky: true

date: 2022-11-19
last_modified_at: 2022-11-19
---

- 처음 블록은 Conv 또는 Conv + activation layer만 존재하는 것 같다.(LeakyReLU)
- 보통 Activation function은 LeakyReLU(0.2)를 사용한다.
- channel의 수는 512를 넘지 않도록 제한되었다.

# 1. LatentCodesDiscriminator

StyleGAN의 style vector를 fake/real 구분하기 위해서 만들어진 Discriminator다.

- input, output 채널이 같은 Linear layer 들과 activation layer로 구성되어 있다.
- 마지막에는 fake/real score를 계산하기 위해서 1 채널로 만들어주는 Linear layer가 있다.
    - Normalization을 사용하지 않는다.
    - 결과 값의 범위를 지정해 주기 위한 activation function을 사용하는 경우가 있다.
- Code
    
    ```python
    class LatentCodesDiscriminator(nn.Module):
        def __init__(self, style_dim=512, n_mlp=4):
            super().__init__()
    
            self.style_dim = style_dim
    
            layers = []
            for i in range(n_mlp-1):
                layers.append(
                    nn.Linear(style_dim, style_dim)
                )
                layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Linear(512, 1))
            self.mlp = nn.Sequential(*layers)
    
        def forward(self, w):
            return self.mlp(w)
    ```
    

# 2. 2D Discrminator

이미지를 input으로 받는 일반적인 2D Discriminator 모델이다. 몇개의 down block(Conv, norm, activation)으로 구성되어 있다. 마지막은 input image가 real/fake score를 계산하도록 channel 수를 1로 지정하였다.

- 각 block을 지나갈 때마다 feature map을 output 으로 내보낸다.
    - loss계산할 때 feature map 끼리도 계산하면 좋아보임
- Code
    
    ```python
    class Discriminator(nn.Module):
        def __init__(self, input_nc=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
            super(Discriminator, self).__init__()
    
            kw = 4
            padw = 1
            self.down1 = nn.Sequential(
                nn.Conv2d(input_nc, 64, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)
            )
            self.down2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=kw, stride=2, padding=padw),
                norm_layer(128), nn.LeakyReLU(0.2, True)
            )
            self.down3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=kw, stride=2, padding=padw),
                norm_layer(256), nn.LeakyReLU(0.2, True)
            )
            self.down4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=kw, stride=2, padding=padw),
                norm_layer(512), nn.LeakyReLU(0.2, True)
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=kw, stride=1, padding=padw),
                norm_layer(512),
                nn.LeakyReLU(0.2, True)
            )
    
            if use_sigmoid:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()
                )
            else:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)
                )
    
        def forward(self, input):
            out = []
            x = self.down1(input)
            out.append(x)
            x = self.down2(x)
            out.append(x)
            x = self.down3(x)
            out.append(x)
            x = self.down4(x)
            out.append(x)
            x = self.conv1(x)
            out.append(x)
            x = self.conv2(x)
            out.append(x)
            
            return out
    ```
    

# 3. NLayerDiscriminator

n개의 Conv block을 자동으로 쌓은 네트워크를 만들어 준다. 구조 자체는 2D Discrmininator와 흡사하다.

- Code
    
    ```python
    class NLayerDiscriminator(nn.Module):
        def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, getIntermFeat=False):
            super(NLayerDiscriminator, self).__init__()
            self.getIntermFeat = getIntermFeat
            self.n_layers = n_layers
    
            kw = 4
            padw = int(np.ceil((kw-1.0)/2))
            sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
    
            nf = ndf
            for n in range(1, n_layers):
                nf_prev = nf
                nf = min(nf * 2, 512)
                sequence += [[
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf), nn.LeakyReLU(0.2, True)
                ]]
    
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]
    
            sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
    
            if use_sigmoid:
                sequence += [[nn.Sigmoid()]]
    
            if getIntermFeat:
                for n in range(len(sequence)):
                    setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
            else:
                sequence_stream = []
                for n in range(len(sequence)):
                    sequence_stream += sequence[n]
                self.model = nn.Sequential(*sequence_stream)
    
        def forward(self, input):
            if self.getIntermFeat:
                res = [input]
                for n in range(self.n_layers+2):
                    model = getattr(self, 'model'+str(n))
                    res.append(model(res[-1]))
                return res[1:]
            else:
                return self.model(input)
    ```
    

## 3.1. setattr

object에 존재하는 속성의 값을 바꾸거나, 새로운 속성을 생성하여 값을 부여한다.

새로운 속성을 생성하여 값을 할당할 수 있다. (새로운 변수 만들기도 가능)

```python
setattr(object,'method',new_value)
```

- Ex
    
    ```python
    class sample:
    	def __init__(self,x):
    			self.x = x
    c = sampel(1)
    print(c.x) # 1
    
    setattr(c,'x',2)
    print(c.x) # 2
    
    setattr(c,'y',3)
    print(c.y) # 3
    ```
    

### In NLayerDiscriminator Code

```python
if getIntermFeat:
  for n in range(len(sequence)):
    setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
```

sequence에 저장되어있는 각 Conv block들에게 model{num} 이라는 변수를 object에 만들어 준다.

- Reference
    
    [Python - setattr(), object의 속성(attribute) 값을 설정하는 함수](https://technote.kr/248)
    

## 3.2. getattr

object에 존재하는 속성 값을 반환한다. 사실상 {object}.{method}와 동일한 기능을 가지고 있다.

```python
getattr(object,'mothod')
```

- Ex
    
    ```python
    class sample:
    	def __init__(self,x):
    			self.x = x
    c = sampel(1)
    print(c.x) # 1
    
    getattr(c,'x')
    print(c.x) # 1
    ```
    

### In NLayerDiscriminator Code

Conv block들을 불러내고 model로 임시 정의한 뒤 forward를 매 block마다 진행하고, 중간 feature map결과물을 res list에 저장한다.

```python
if self.getIntermFeat:
  res = [input]
  for n in range(self.n_layers+2):
      model = getattr(self, 'model'+str(n))
      res.append(model(res[-1]))
```

- Reference
    
    이 외에도 hasattr, delattr 가 있으니 확인해 보자
    
    [Python - getattr(), object의 속성(attribute) 값을 확인하는 함수](https://technote.kr/249)
    

# 4. MultiscaleDiscriminator

동일한 구조의 여러 Discriminator를 만들고 (NLayerDiscriminator를 이용한다.) input의 size를 AvgPool로   조절하여 크기별로 Discriminator에 전달하여 결과물을 얻는다. Discriminator는 크기에 따른 input image에 대해서 real/fake를 구분하게 된다.

- Code
    
    ```python
    class MultiscaleDiscriminator(nn.Module):
        def __init__(self, input_nc=3, ndf=64, n_layers=6, norm_layer=nn.InstanceNorm2d,
                     use_sigmoid=False, num_D=3, getIntermFeat=False):
            super(MultiscaleDiscriminator, self).__init__()
            self.num_D = num_D
            self.n_layers = n_layers
            self.getIntermFeat = getIntermFeat
    
            for i in range(num_D):
                netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
                if getIntermFeat:
                    for j in range(n_layers + 2):
                        setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
                else:
                    setattr(self, 'layer' + str(i), netD.model)
    
            self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
    
        def singleD_forward(self, model, input):
            if self.getIntermFeat:
                result = [input]
                for i in range(len(model)):
                    result.append(model[i](result[-1]))
                return result[1:]
            else:
                return [model(input)]
    
        def forward(self, input):
            num_D = self.num_D
            result = []
            input_downsampled = input
            for i in range(num_D):
                if self.getIntermFeat:
                    model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                             range(self.n_layers + 2)]
                else:
                    model = getattr(self, 'layer' + str(num_D - 1 - i))
                result.append(self.singleD_forward(model, input_downsampled))
                if i != (num_D - 1):
                    input_downsampled = self.downsample(input_downsampled)
            return result
    ```
    

# 5. StarGANv2Discriminator

몇개의 resblock, Conv block 을 가지고 있는 Discriminator다. 여러 resblock을 가진 구조이기 때문에 GPU memory를 많이 먹는다.

- 반복되는 ResBlk의 개수는 `int(np.log2(img_size)) - 2` 로 설정되었다.
- Code
    
    ```python
    class StarGANv2Discriminator(nn.Module):
        def __init__(self, img_size=256, max_conv_dim=512):
            super().__init__()
            dim_in = 2**14 // img_size
    
            blocks = []
            blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
    
            repeat_num = int(np.log2(img_size)) - 2
    
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
    
            blocks += [nn.LeakyReLU(0.2)]
            blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
            blocks += [nn.LeakyReLU(0.2)]
            blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
            self.main = nn.Sequential(*blocks)
    
        def forward(self, x):
            out = self.main(x)
            out = out.view(out.size(0), -1)  # (batch, num_domains)
            return out
    ```