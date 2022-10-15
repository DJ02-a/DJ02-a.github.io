---
title: "tqdm"
excerpt: "tqdm의 사용법 및 간단한 옵션 소개"

categories:
  - Python
  - tqdm

tags:
  - [tqdm]

permalink: /Python/tqdm/

toc: true
toc_sticky: true

date: 2022-07-29
last_modified_at: 2022-10-01
---


# 1. tqdm

---

작업 진행 상황을 표시하기 위한 도구.

iteration 가능한 변수를 tqdm() 안에 넣어주면 진행 상황을 terminal에 표시한다.

```python
from tqdm import tqdm

numbers = ['1','2','3','4']
pbar = tqdm(numbers)
for i in pbar:
	print(i)
```

## 1.1. tqdm options

tqdm class를 call 할 때 tqdm parameter를 설정할 수 있다.

```python
from tqdm import tqdm
numbers = ['1','2','3','4']
pbar = tqdm(numbers, desc='tqdm example', mininterval=0.01)
```

<img src="/assets/images/posts_img/2022-07-29-Python-tqdm/1.pbar.png">

- desc(str) : 진행 바 앞에 텍스트 출력
- total(int) : 전체 반복량
- ncols : 진행바 컬럼 길이
- miniterval, maxinterval : 업데이트 주기 sec 단위다.
- ascii(bool) : ‘#’ 문자로 진행바가 표시
- initial : 진행 시작 값

루프를 도는 중에 수동 tqdm 인스턴스 선언이 가능하다.

```python
pbar = tqdm([1,2,3,4])
for i in pbar:
	pbar.set_description(f'{i}')
```

- pbar.set_description() : 괄호 안의 내용을 bar 옆에 표시해 준다.
- pbar.update(10) : bar 진행 상황을 수동으로 변경 가능하다.