---
title: "Decorator"
excerpt: "Decorator"

categories:
  - Python
  - Decorator

tags:
  - [Decorator]

permalink: /Python/Decorator/

toc: true
toc_sticky: true

date: 2022-07-05
last_modified_at: 2022-07-05
---

# 1. Decorator 란

- class에서 method를 정의할 때 위에 `@staticmethod, @classmethod, @abstractmethod` 처럼 `@`으로 시작하는 것들이 Decorator다.
- 함수를 수정하지 않고도 유연하게 함수에 특정 동작을 추가하거나 작동 방식을 바꿀 수 있다.
- closure와 다르게 함수를 다른 함수의 인자로 전달한다는 점이 다르다.
    - closure는 변수를 인자로 넘겨준다.
    

일반적으로 데코레이터는 로그를 남기거나 유저의 로그인 상태 등을 확인하여 로그인 상태가 아니면 로그인 페이지로 redirect하기 위해서 많이 사용됩니다. 또한 프로그램의 성능을 테스트하기 위해서도 많이 쓰입니다. 리눅스나 유닉스 서버 관리자는 스크립트가 실행되는 시간을 측정하기 위해서 다음과 같은 date와 time 명령어를 많이 사용합니다.

# 2. 매개변수를 함수 이름으로 받는 경우

## 2.1. Decorator를 사용하지 않는 경우

- Code
    
    함수의 실행 상황을 추적할 때 ‘trace’ 라 한다.
    
    함수를 생성하면서 어떤 함수가 생성되었는지 추적 가능
    
    ```python
    def trace(func):                             # 호출할 함수를 매개변수로 받음
        def wrapper():                           # 호출할 함수를 감싸는 함수 ???
            print(func.__name__, '함수 시작')    # __name__으로 함수 이름 출력 
            func()                               # 매개변수로 받은 함수를 호출
            print(func.__name__, '함수 끝')
        return wrapper                           # wrapper 함수 반환
     
    def hello():
        print('hello')
     
    def world():
        print('world')
     
    # 뭔가 여러번 호출해야 해서 불편하다...
    trace_hello = trace(hello)    # 데코레이터에 호출할 함수를 넣음, trace의 return 값은 함수
    trace_hello()                 # 반환된 함수를 호출
    trace_world = trace(world)    # 데코레이터에 호출할 함수를 넣음
    trace_world()                 # 반환된 함수를 호출
    ```
    

## 2.2. Decorator를 사용하는 경우

`@trace` 는 위에서 trace 함수를 대신한다.

- Code
    
    ```python
    def trace(func):                             # 호출할 함수를 매개변수로 받음
        def wrapper():
            print(func.__name__, '함수 시작')    # __name__으로 함수 이름 출력
            func()                               # 매개변수로 받은 함수를 호출
            print(func.__name__, '함수 끝')
        return wrapper                           # wrapper 함수 반환
    
    @trace    # @데코레이터
    def hello():
        print('hello')
     
    @trace    # @데코레이터
    def world():
        print('world')
     
    hello()    # 함수를 그대로 호출
    world()    # 함수를 그대로 호출
    ```
    

이렇게 데코레이터는 함수를 매개변수로 가지며, 함수를 불러옴과 동시에 추가 기능을 구현할 때 사용된다.

## 2.3. 예시

- Code
    - decorator_function을 반복적으로 불러온다.
    
    ```python
    def decorator_function(original_function):
        def wrapper_function():
            print('{} 함수가 호출되기전 입니다.'.format(original_function.__name__))
            return original_function()
    
        return wrapper_function
    
    def display_1():
        print('display_1 함수가 실행됐습니다.')
    
    def display_2():
        print('display_2 함수가 실행됐습니다.')
    
    display_1 = decorator_function(display_1)  # 1
    display_2 = decorator_function(display_2)  # 2
    
    display_1()
    print()
    display_2()
    ```
    
    - Decoder를 사용하면 자동으로 진행
    - 코드가 더 깔끔해진다.
    
    ```python
    def decorator_function(original_function):
        def wrapper_function():
            print('{} 함수가 호출되기전 입니다.'.format(original_function.__name__))
            return original_function()
    
        return wrapper_function
    
    @decorator_function  # 1
    def display_1():
        print('display_1 함수가 실행됐습니다.')
    
    @decorator_function  # 2
    def display_2():
        print('display_2 함수가 실행됐습니다.')
    
    # display_1 = decorator_function(display_1)  # 3
    # display_2 = decorator_function(display_2)  # 4
    
    display_1()
    print()
    display_2()
    ```
    

# 3. 인수를 가진 함수를 데코레이팅하고 싶을 때

인수를 가진 함수를 데코레이팅 하기 위해서는 *args, **kwargs 를 추가한다.

## 3.1. *args, **kwargs 란

1. *args 를 이용하면 key, value를 받는 형식을 제외한 모든 형식을 인자로 받을 수 있다.
2. **kwarg 를 사용하면 key, value를 받는 형식의 인자를 받을 수 있다.
    - 사실상 dictionary 형태면 **kwarg으로 넘겨지는 듯
3. *args 나 **kwarg 중 어느 것을 쓰든 받는 인자 양식이 받으면 여러 개의 인자를 받을 수 있다.
4. *args, **kwarg 를 같이 사용하면 어떤 형태의 인자이든 다 받겠다는 의미이다.
5. 4번 사용중 제한점은 순서는 지켜져야 한다는 것이다. 가령 아래 코드처럼 작성하면 오류가 발생한다.
    
    ```python
    a_func({"myname",'kc'},"hi")
    ```
    
- Reference
    
    [파이썬 *args, **kwargs 의미와 예제를 통해 이해하기](https://scribblinganything.tistory.com/161)
    

- Code
    
    ```python
    def decorator_function(original_function):
        def wrapper_function(*args, **kwargs):  #1
            print('{} 함수가 호출되기전 입니다.'.format(original_function.__name__))
            return original_function(*args, **kwargs)  #2
        return wrapper_function
    
    @decorator_function
    def display():
        print('display 함수가 실행됐습니다.')
    
    @decorator_function
    def display_info(name, age):
        print('display_info({}, {}) 함수가 실행됐습니다.'.format(name, age))
    
    display()
    print()
    display_info('John', 25)
    
    # $ python decorator.py
    # display 함수가 호출되기전 입니다.
    # display 함수가 실행됐습니다.
    # 
    # display_info 함수가 호출되기전 입니다.
    # display_info(John, 25) 함수가 실행됐습니다.
    ```
    

# 4. Class 형식으로 지정하는 Decorator

Decorator를 class로 만들 때는 instance를 만들지 않아도 사용할 수 있다.

- Code
    
    ```python
    class DecoratorClass:  # 1
        def __init__(self, original_function):
            self.original_function = original_function
    
        def __call__(self, *args, **kwargs):
            print('{} 함수가 호출되기전 입니다.'.format(self.original_function.__name__))
            return self.original_function(*args, **kwargs)
    
    @DecoratorClass  # 2
    def display():
        print('display 함수가 실행됐습니다.')
    
    @DecoratorClass  # 3
    def display_info(name, age):
        print('display_info({}, {}) 함수가 실행됐습니다.'.format(name, age))
    
    display()
    print()
    display_info('John', 25)
    ```
    

# 5. 여러 개 Decorator 지정

```python
# 데코레이터가 실행되는 순서는 위에서 아래 순이다.
@데코레이터1
@데코레이터2
def 함수이름():
    코드
```

- Code
    
    ```python
    import datetime
    import time
    
    def my_logger(original_function):
        import logging
        filename = '{}.log'.format(original_function.__name__)
        logging.basicConfig(handlers=[logging.FileHandler(filename, 'a', 'utf-8')],
                            level=logging.INFO)
    
        def wrapper(*args, **kwargs):
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            logging.info(
                '[{}] 실행결과 args - {}, kwargs - {}'.format(timestamp, args, kwargs))
            return original_function(*args, **kwargs)
    
        return wrapper
    
    def my_timer(original_function):
        import time
    
        def wrapper(*args, **kwargs):
            t1 = time.time()
            result = original_function(*args, **kwargs)
            t2 = time.time() - t1
            print('{} 함수가 실행된 총 시간: {} 초'.format(original_function.__name__, t2))
            return result
    
        return wrapper
    
    @my_logger  # 1
    @my_timer  # 2
    def display_info(name, age):
        time.sleep(1)
        print('display_info({}, {}) 함수가 실행됐습니다.'.format(name, age))
    
    display_info('John', 25)
    
    # $ python decorator.py
    # display_info(John, 25) 함수가 실행됐습니다.
    # display_info 함수가 실행된 총 시간: 1.00419592857 초
    ```
    

# 6. Example Code

```python
import datetime
import time

def my_logger(original_function):
    import logging
    filename = '{}.log'.format(original_function.__name__)
    logging.basicConfig(handlers=[logging.FileHandler(filename, 'a', 'utf-8')],
                        level=logging.INFO)

    def wrapper(*args, **kwargs):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        logging.info(
            '[{}] 실행결과 args - {}, kwargs - {}'.format(timestamp, args, kwargs))
        return original_function(*args, **kwargs)

    return wrapper

def my_timer(original_function):  # 1
    import time

    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = original_function(*args, **kwargs)
        t2 = time.time() - t1
        print('{} 함수가 실행된 총 시간: {} 초'.format(original_function.__name__, t2))
        return result

    return wrapper

@my_timer  # 2
def display_info(name, age):
    time.sleep(1)
    print('display_info({}, {}) 함수가 실행됐습니다.'.format(name, age))

display_info('John', 25)
```

# 7. Reference
    
[파이썬 코딩 도장](https://dojang.io/mod/page/view.php?id=2427)

[파이썬 - 데코레이터 (Decorator) - schoolofweb.net](https://schoolofweb.net/blog/posts/%ed%8c%8c%ec%9d%b4%ec%8d%ac-%eb%8d%b0%ec%bd%94%eb%a0%88%ec%9d%b4%ed%84%b0-decorator/)
    