---
title: "ABC(Abstract Base Class)"
excerpt: "ABC의 사용법 및 code, 간단한 옵션 소개"

categories:
  - Python
  - ABC

tags:
  - [ABC]

permalink: /Python/ABC/

toc: true
toc_sticky: true

date: 2022-10-01
last_modified_at: 2022-10-01
---

# 1. ABC 란

Base class를 상속받는 파생 class가 반드시 Base class의 method를 반드시 명시적으로 선언해서 구현하도록 강제한다.

- 상속(Inheritance) : 클래스의 재 사용성을 높여서 코드의 반복에 따른 유지 보수 비용을 낮춰준다.
- 다형성(Polymorphism) : 하나의 인터페이스(Base class)를 통해 서로 다른 여러 파생 class를 제공한다.
    - 파생 class를 만들 때 직접 작성이 필요한 method를 지정하기 위해서 이용. 강제성을 부여

# 2. Python Code

추상화 시키고자 하는 method에 데코레이터로 `@abstractmethod` 를 선언해 주면 된다.

- Code
    
    ```python
    import abc
    class BaseClass:
    
    	__metaclass__ = abc.ABCMeta
    
    	@abc.abstractmethod
    	def func1(self):
    		pass
    
    	@abc.abstractmethod
    	def func2(self):
    		pass
    ```
    

# 3. 특징

1. abc 클래스를 이용하게 되면, 적용된 BaseClass는 instance화 되지 못한다.
    - Code
        
        ```python
        from BaseClass import BaseClass
        
        base = BaseClass()
        >> TypeError
        ```
        
2. 에러 메세지 발생 지점이 다르다.
abc 클래스를 사용하지 않은 경우 instance에서 선언되지 않은 method 호출시에 에러를 발생시킨다.
abc 클래스를 사용하는 경우 상속한 class를 import / instance 화 할 때 에러를 발생시킨다.
    - Code
        
        ```python
        >>> from BaseClass import BaseClass
        >>> 
        >>> base = BaseClass()
        Traceback (most recent call last):  
        File "<stdin>", line 1, in <module>
        TypeError: Can't instantiate abstract class BaseClass with abstract methods func1, func2
        ```
        
# 4. Reference
    
[Python ABC(Abstract Base Class) 추상화 클래스](https://bluese05.tistory.com/61)
    
