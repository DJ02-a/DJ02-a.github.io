---
title: "Magic Method"
excerpt: "Magic Method"

categories:
  - Python
  - Magic Method

tags:
  - [Magic Method]

permalink: /Python/Magic-Method/

toc: true
toc_sticky: true

date: 2022-08-18
last_modified_at: 2022-08-18
---

# 1. Magic method 란?

---

- class 안에 정의할 수 있는 스페셜 method이며 class를 int, str, list등의 파이썬의 빌트인 타입(built-in type) 과 같은 작동을 하게 해준다.
- + , -, >, < 등의 오퍼레이터에 대해서 각각의 데이터 타입에 맞는 메소드로 오버로딩하여 백그라운드에서 연산한다.
- __init__ 이나 __str__ 과 같이 메소드 이름 앞뒤에 더블 언더스코어(”__”)를 붙인다.
(일반적인 호칭은 “던더 init 던더” 라 한다.)

- Think
    - 알고 보니 우리가 사용하는 dtype들은 전부 class를 통해 정의되는 것이다.
    - 예를 들어 5라는 숫자를 변수로 지정하면 int라는 class에 자동으로 넣어주어 instance가 생성되고, 이 변수는 int class의 magic method를 이용할 수 있게 된다.
    - my_num + 5 == my_num.__add__(5) 라고 하지만 저런 operator가 어떻게 자동으로 더해주는지 궁금
        - 뭔가 operator와 magic method가 서로 대응되는 함수가 있는 듯..?

## 1.1. __init__, __str__

클래스를 만들 때 항상 사용하는 __init__,  __str__은 가장 대표적인 매직 메소드이다.

```python
# -*- coding: utf-8 -*-

class Dog(object):
    def __init__(self, name, age):
        print('이름: {}, 나이: {}'.format(name, age))

dog_1 = Dog('Pink', '12')

$ python oop_6.py
이름: Pink, 나이: 12
```

- __init__ : 은 class에서 instance를 생성할 때 자동으로 실행되는 매직 메소드다.

```python
# -*- coding: utf-8 -*-

class Food(object):
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def __str__(self):
        return '아이템: {}, 가격: {}'.format(self.name, self.price)

food_1 = Food('아이스크림', 3000)

# 인스턴스 출력
print(food_1)
```

- __str__ : 은 instance를 그대로 출력할 때 자동으로 실행되는 매직 메소드다. 일반적으로 instance가 저장된 메모리 주소가 출력 된다.

## 1.2. +, - operator

사실 자주 사용하는 +, - 또한 매직 메소드를 호출하는 오퍼레이터이다.

x + y를 실행하면 사실 “__add__”가 호출되어 백그라운드에서는 x.__add__(y) 가 실행되는 형식이다.

```python
# -*- coding: utf-8 -*-

# int를 부모 클래스로 가진 새로운 클래스 생성
class MyInt(int):
    pass

# 인스턴스 생성
my_num = MyInt(5)

# 타입 확인
print(type(my_num))  # => <class '__main__.MyInt'>

# int의 인스턴스인지 확인
print(isinstance(my_num, int))  # => True

# MyInt의 베이스 클래스 확인
print(MyInt.__bases__)  # => (<type 'int'>,)

print(MyInt + 5) # => 10
```

> int는 class다.
MyInt가 int를 상속받으므로 int처럼 사용 가능하다.
> 

    

## 1.3. Magic method 체크

class 로 부터 생성된 instance의 magic method를 체크하는 방법이다.

```python
# -*- coding: utf-8 -*-

# int를 부모 클래스로 가진 새로운 클래스 생성
class MyInt(int):
    pass

# 인스턴스 생성
my_num = MyInt(5)

print(dir(my_num))
"""
['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dict__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']
"""

# 매직 메소드를 직접 호출
print(mynum.__add__(5)) # => 10 / mynum + 5 와 동일한 의미를 지닌다.
```

class는 기본적으로 다양한 magic method를 가지고 있다.

- magic methods
    
    `'__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__'`
    

## 1.4. Magic method 수정하기

int class를 상속 받은 MyInt class에서 __add__가 수정 된다.

```python
# -*- coding: utf-8 -*-

# int를 부모 클래스로 가진 새로운 클래스 생성
class MyInt(int):
    # __add__ 변경
    def __add__(self, other):
        return '{} 더하기 {} 는 {} 입니다'.format(self.real, other.real, self.real + other.real)

# 인스턴스 생성
my_num = MyInt(5)

print(my_num + 5)  # => 5 더하기 5 는 10 입니다
```

## 1.5. “<” 연산 megic method

__lt__ 메소드를 수정하여 비교를 할 수 있다.

```python
# -*- coding: utf-8 -*-

class Food(object):
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def __lt__(self, other):
        if self.price < other.price:
            return True
        else:
            return False

food_1 = Food('아이스크림', 3000)
food_2 = Food('햄버거', 5000)
food_3 = Food('콜라', 2000)

# food_2가 food_1보다 큰지 확인
print(food_1 < food_2)  # 3000 < 5000
print(food_2 < food_3)  # 5000 < 2000-
```

# 2. Reference
    
[[파이썬] __eq__ __lt__ 함수에 대해](https://darkstart.tistory.com/180)
    
[파이썬 - OOP Part 6. 매직 메소드 (Magic Method) - schoolofweb.net](https://schoolofweb.net/blog/posts/%ed%8c%8c%ec%9d%b4%ec%8d%ac-oop-part-6-%eb%a7%a4%ec%a7%81-%eb%a9%94%ec%86%8c%eb%93%9c-magic-method/)