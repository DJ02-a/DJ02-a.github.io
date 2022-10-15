---
title: "Generator"
excerpt: "Generator"

categories:
  - Python
  - Generator

tags:
  - [Generator]

permalink: /Python/Generator/

toc: true
toc_sticky: true

date: 2022-08-02
last_modified_at: 2022-08-02
---

---

일반함수가 호출되면 코드의 첫 번째 행으로부터 시작하여 리턴(return), 예외(exception) 또는 마지막 줄까지 실행된 후, 모든 컨트롤을 리턴한다. 그리고 함수가 가지고 있던 모든 내부 함수나 모든 로컬 변수는 메모리상에서 사라진다.

그러나 프로그래머들은 한번에 모든 일을 하고 사라지는 일반 함수가 아니라 하나의 일을 마치면 자기가 하고 있던 일을 기억하면서 대기하고 있다가 다시 호출되면 전의 일을 계속 이어서 하는 똑똑한 함수를 만들었다. 그것이 generator이다. 모든 결과값을 메모리에 저장하지 않기 때문에 메모리 효율이 느는 장점이 있다.

# 1. yield 키워드

일반적으로 함수는 어떤 결과 값을 `return` 키워드를 이용해서 반환한다. 하지만 `yield` 키워드를 이용해서 다소 **다른 방식**으로 결과 값을 제공할 수 있다.

yield 키워드는 결과값을 여러번 나누어서 제공한다는 특징이 있다.

- Code
    
    ```python
    def return_abc():
      return list("ABC")
    
    def yield_abc():
      yield "A"
      yield "B"
      yield "C"
    
    for ch in return_abc():
      print(ch)
    
    # A
    # B
    # C
    
    for ch in yield_abc():
      print(ch)
    
    # A
    # B
    # C
    # 결과는 동일해 보임
    ```
    

함수를 호출한 결과 값을 바로 출력해본다.

```python
>>> print(return_abc())
['A', 'B', 'C']
>>> print(yield_abc())
<generator object yield_abc at 0x7f4ed03e6040>
```

결론적으로 `yield` 키워드를 사용하면 제너레이터를 반환한다.

# 2. Generator

```python
def square_numbers(nums):
    for i in nums:
        yield i * i

my_nums = square_numbers([1, 2, 3, 4, 5])  #1

print(my_nums)

# $ python generator.py
# <generator object square_numbers at 0x0000016B17E19EB0>

print(next(my_nums))
print(next(my_nums))
print(next(my_nums))

# $ python generator.py
# 1
# 4
# 9
```

class의 instance를 만들어 주는 것처럼 한번 generator를 선언해 주고, `next({generator})` 를 통해서 yield 에서 return되는 값을 순차적으로 가져온다. 만약 함수에 존재하는 `yield` 수 보다 `next`로 호출하는 수가 더 많으면 error가 발생한다. 

일반적으로 generator는 for문을 이용하여 사용하기도 한다.

```python
def square_numbers(nums):
    for i in nums:
        yield i * i

my_nums = square_numbers([1, 2, 3, 4, 5])

for num in my_nums:
    print(num)

# $ python generator.py
# 1
# 4
# 9
# 16
# 25
```

next를 사용하지 않고 for문을 사용하는 경우 error가 발생하지 않고 안전하게 반환된다.

모든 데이터를 반드시 한번에 처리하거나 얻지 않아도 될 때 이용하면 유용할 듯 하다. 특히 **`generator(num_people)` 와 같이 굳이 한번에 모든 데이터를 만들지 않고 call 할때만 데이터를 만들어도 되는 경우 유용하게 쓰일듯 하다. 

실행 시간 보다 메모리 소비를 줄여야 하는 경우라면 제너레이터를 사용해야하고, 리소스 보다 실행 시간을 줄여야 하는 경우라면 리스트를 사용해야 한다고 볼 수 있다. 

# 3. Reference
    
[파이썬 - 제너레이터 (Generator) - schoolofweb.net](https://schoolofweb.net/blog/posts/%ed%8c%8c%ec%9d%b4%ec%8d%ac-%ec%a0%9c%eb%84%88%eb%a0%88%ec%9d%b4%ed%84%b0-generator/)

[파이썬의 yield 키워드와 제너레이터(generator)](https://www.daleseo.com/python-yield/)

