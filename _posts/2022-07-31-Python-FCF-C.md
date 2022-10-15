---
title: "First Class Function & Closure"
excerpt: "First Class Function & Closure"

categories:
  - Python
  - First Class Function
  - Closure
  - First Class Function Closure

tags:
  - [First-Class-Function-Closure]

permalink: /Python/First-Class-Function-Closure/

toc: true
toc_sticky: true

date: 2022-07-31
last_modified_at: 2022-07-31
---

# 1. First Class Function

---

- 변수에 함수를 할당할 수 있다.
    - 변수에는 함수가 있는 메모리 주소가 저장된다.
    - 함수를 할당할 때 매개변수를 넣어줘야 한다.
    - 변수에 함수를 할당할 때 return 직전 줄 까지 진행된다.
    - `변수()`를 통해서 메모리 주소로 함수를 호출한다.(작동 시작)
    - `변수()`로 return 줄을 실행할 수 있다.
- 인자로써 다른 함수에 전달할 수 있다.
- 함수를 리턴에 사용할 수 있다.

## 1.1. 변수에 함수 할당하기

python 언어가 퍼스트 클래스 함수를 지원하기 때문에 가능하다.

```python
def square(x):
    return x * x

f = square

print(f(5))
```

## 1.2. 함수를 매개변수로 사용하기

- 함수를 변수에 할당하지 않고 바로 매개변수로 사용할 수 있다.

```python
def square(x):
    return x * x

def my_map(func, arg_list):
    result = []
    for i in arg_list:
        result.append(func(i)) # square 함수 호출, func == square
    return result

num_list = [1, 2, 3, 4, 5]

squares = my_map(square, num_list)

print(squares)

# result
# [1, 4, 9, 16, 25]
```

그러나 단순히 list 값을 제곱 하는 방법은 `square` 함수를 사용하지 않고도 가능하다. 

- Code
    
    ```python
    def square(x):
        return x * x
    
    num_list = [1, 2, 3, 4, 5]
    
    def simple_square(arg_list):
        result = []
        for i in arg_list:
            result.append(i * i)
        return result
    
    simple_squares = simple_square(num_list)
    
    print(simple_squares)
    ```
    

하지만 퍼스트 클래스 함수를 사용하면 이미 정의된 여러 함수를 간단히 재활용 가능하다.

- square function 관점
    
    `square` 함수를 `my_map`함수 말고 여러 곳에서 사용한다면 첫번째 코드처럼 `square` 함수를 따로 정의하여 필요할 때 부르는 것이 좋아보인다.
    
- my_map function 관점
    
    my_map에는 func의 결과물들을 result list에 저장하는 코드를 지니고 있다. 원하는 func의 결과물을 저장하고 싶다면, 매개변수를 함수로 받아 사용하는 것이 좋아보인다.
    

## 1.3. 리턴값을 함수로 사용하기(closure)

- 일반적으로 msg와 같은 함수의 지역변수 값은 함수가 호출된 이후에 메모리상에서 사라지므로 다시 참조할 수 없다.
- 그러나 `return` 시 사용된 함수(log_message)는 다른 함수의 지역변수(logger의 msg)를 그 함수(logger)가 종료된 이후에도 기억할 수 있다.

**추후 다른 페이지에서 자세하게 다룬다.**

```python
def logger(msg):
    def log_message():  # 1
        print('Log: ', msg)

    return log_message

log_hi = logger('Hi')
print(log_hi)  # log_message 오브젝트가 출력됩니다.
log_hi()  # "Log: Hi"가 출력됩니다.

# result
#<function logger.<locals>.log_message at 0x0000022AB43EAA60>
#Log:  Hi

del logger  # 글로벌 네임스페이스에서 logger 오브젝트를 지웁니다.

# logger 오브젝트가 지워진 것을 확인합니다.
try:
    print(logger)
except NameError:
    print('NameError: logger는 존재하지 않습니다.')

log_hi()  # logger가 지워진 뒤에도 Log: Hi"가 출력됩니다.

# result
# <function logger.<locals>.log_message at 0x0000022EC0BBAAF0>
# Log:  Hi
# NameError: logger는 존재하지 않습니다.
# Log:  Hi
```

## 1.4. 예시

- Code1
    
    ```python
    # 단순한 일반 함수
    def simple_html_tag(tag, msg):
        print('<{0}>{1}<{0}>'.format(tag, msg))
    
    simple_html_tag('h1', '심플 헤딩 타이틀')
    
    print('-' * 30)
    
    # 함수를 리턴하는 함수
    def html_tag(tag):
        def wrap_text(msg):
            print('<{0}>{1}<{0}>'.format(tag, msg))
    
        return wrap_text
    
    print_h1 = html_tag('h1')  # 1
    print(print_h1)  # 2
    print_h1('첫 번째 헤딩 타이틀')  # 3
    print_h1('두 번째 헤딩 타이틀')  # 4
    
    print_p = html_tag('p')
    print_p('이것은 패러그래프 입니다.')
    
    # result
    # <h1>심플 헤딩 타이틀<h1>
    # ------------------------------
    # <function html_tag.<locals>.wrap_text at 0x00000272C3CFAAF0>
    # <h1>첫 번째 헤딩 타이틀<h1>
    # <h1>두 번째 헤딩 타이틀<h1>
    # <p>이것은 패러그래프 입니다.<p>
    ```
    
- Code2
    
    ```python
    # 함수를 리턴하는 함수
    def html_tag(tag):
        def wrap_text(msg):
            print('2')
            print('<{0}>{1}<{0}>'.format(tag, msg))
        print('1')
        return wrap_text
    
    method = html_tag('0')
    print(method)
    method('testing...')
    
    # result
    # 1
    # <function html_tag.<locals>.wrap_text at 0x7f1d6e0f8d30>
    # 2
    # <0>testing...<0>
    ```

# 2. Closure

- outer function에서 return하는 inner function이 존재하는 경우 inner function 내에 outer function의 지역변수가 남아있다.
- closure가 있는 outer function을 선언하면 `__closure__` 라는 magic method가 생성된다.(다른 메모리 주소에 저장된다.)
- `__closure__` 에는 inner function에서 사용된 outer_func의 변수 값이 기록된 메모리 주소들이 튜플로 묶여있다.
- 변수 값을 보기 위해서는 `outer_fun.__closure__[idx].cell_contents` 을 이용한다.
    - 변수 명을 보기 위해서 어떤걸 써야하는지 모르겠다;;
- Code
    
    ```python
    def outer_func():  # 1
        message = 'Hi'  # 3
    
        def inner_func():  # 4
            print(message)  # 6
    
        return inner_func  # 5
    
    my_func = outer_func()  # 2
    
    print(my_func)  # 7
    print()
    print(dir(my_func))  # 8
    print()
    print(type(my_func.__closure__))  # 9
    print()
    print(my_func.__closure__)  # 10
    print()
    print(my_func.__closure__[0])  # 11
    print()
    print(dir(my_func.__closure__[0]))  # 12
    print()
    print(my_func.__closure__[0].cell_contents)  # 13
    ```
    

## 2.1. Code Test

- outer function의 return에 inner function이 존재하지 않는 경우 outer function에 `__closure__` megic method는 존재하지 않는다.
- 다수의 inner function을 outer function의 return에 넣어줘도 `__closure__` megic method는 존재하지 않는다.
    - 반드시 outer function의 return에는 하나의 inner function을 적어야 한다.
- __closer__ 내용을 보기 위해서 밑의 코드를 이용한다.
    
    ```python
    outer_fun.__closure__[0].cell_contents
    ```
    
    이 코드로 outer_fun에 존재하던 변수 값을 부를 수 있다.
    
# 3. Reference
    
[파이썬 - 퍼스트클래스 함수 (First Class Function) - schoolofweb.net](https://schoolofweb.net/blog/posts/%ed%8c%8c%ec%9d%b4%ec%8d%ac-%ed%8d%bc%ec%8a%a4%ed%8a%b8%ed%81%b4%eb%9e%98%ec%8a%a4-%ed%95%a8%ec%88%98-first-class-function/)