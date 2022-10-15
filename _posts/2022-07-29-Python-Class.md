---
title: "Class"
excerpt: "Class"

categories:
  - Python
  - Class

tags:
  - [Class]

permalink: /Python/Class/

toc: true
toc_sticky: true

date: 2022-07-29
last_modified_at: 2022-10-01
---


# 1. Instance method

인스턴스를 통해서 호출이 되고, 첫 번째 인자로 인스턴스 자신을 자동으로 전달한다. 이 인수를 `self` 라 한다. 

데이터를 추가, 수정, 삭제가 적용되는 범위는 해당 instance 내로 한정된다. 공통 class에서 나온 서로 다른 instance 끼리 영향을 미치지 않는다는 의미이다.

# 2. Class method

class method는 ‘cls’인 클래스를 인자로 받고 모든 인스턴스가 공유하는 클래스 변수와 같은 데이터를 생성, 변경 또는 참조하기 위한 메소드라고 생각하면 된다.

사실 그냥 직접 클래스 변수를 변경하는 방법도 있지만, 데이터 검사나 다른 부가 기능 등의 추가가 필요 할 때 class method를 사용하면 매우 편리하다. instance를 먼저 생성하고 classmethod를 호출하지 않아도 된다. instance를 생성함과 동시에 classmethod를 호출해도 된다.

- Code
    
    ```python
    class Employee:
        raise_amount = 1.1  # 연봉 인상율 클래스 변수
    
        def __init__(self, first, last, pay):
            self.first = first
            self.last = last
            self.pay = pay
    
        def apply_raise(self):
            self.pay = int(self.pay * self.raise_amount)
    
        def full_name(self):
            return '{} {}'.format(self.first, self.last)
    
        def get_pay(self):
            return '현재 "{}"의 연봉은 "{}"입니다.'.format(self.full_name(), self.pay)
    
        # 1 클래스 메소드 데코레이터를 사용하여 클래스 메소드 정의
        @classmethod
        def change_raise_amount(cls, amount):
            # 2 인상율이 "1" 보다 작으면 재입력 요청
            while amount < 1:
                print('[경고] 인상율은 "1"보다 작을 수 없습니다.')
                amount = input('[입력] 인상율을 다시 입력하여 주십시오.\n=> ')
                amount = float(amount)
            cls.raise_amount = amount
            print('인상율 "{}"가 적용 되었습니다.'.format(amount))
    
    emp_1 = Employee('Sanghee', 'Lee', 50000)
    emp_2 = Employee('Minjung', 'Kim', 60000)
    
    # 연봉 인상 전
    print(emp_1.get_pay())
    print(emp_2.get_pay())
    
    # 연봉 인상율 변경
    Employee.change_raise_amount(0.9)
    
    # 연봉 인상
    emp_1.apply_raise()
    emp_2.apply_raise()
    
    # 연봉 인상 후
    print(emp_1.get_pay())
    print(emp_2.get_pay())
    ```
    

class method는 instance 생성자와 같은 용도로 사용하는 경우도 있다. 고정된 값의 init input이 아니라 다른 방식으로 init input을 받고 수정하여 instance를 생성할 수 있다.

밑은 return에 cls(자기자신 class)을 호출하여 __init__(self, 변수)을 실행하는 코드다.

- Code 1
    - Data가 주민등록번호로 들어왔을 경우 분석해주는 ssn_parser function을 class 밖에서 정의하였다.
    - Instance를 생성할 때 ssn_parser를 호출하여 데이터를 나누고 class의 __init__을 호출한다.
    - ss_parser는 class와 관련된 함수이기 때문에 class 안에 넣어주는게 보기 좋을 듯 하다.
    
    ```python
    class Person:
        def __init__(self, year, month, day, sex):
            self.year = year
            self.month = month
            self.day = day
            self.sex = sex
    
        def __str__(self):
            return '{}년 {}월 {}일생 {}입니다.'.format(self.year, self.month, self.day, self.sex)
    
    ssn_1 = '900829-1034356'
    ssn_2 = '051224-4061569'
    
    def ssn_parser(ssn):
        front, back = ssn.split('-')
        sex = back[0]
    
        if sex == '1' or sex == '2':
            year = '19' + front[:2]
        else:
            year = '20' + front[:2]
    
        if (int(sex) % 2) == 0:
            sex = '여성'
        else:
            sex = '남성'
    
        month = front[2:4]
        day = front[4:6]
    
        return year, month, day, sex
    
    person_1 = Person(*ssn_parser(ssn_1))
    print(person_1)
    
    person_2 = Person(*ssn_parser(ssn_2))
    print(person_2)
    ```
    
- Code 2
    - @classmethod 를 이용하여 class 안에 넣었다.
    - instance를 생성할 때 바로 __init__을 호출하는게 아니라 ssn_constructor를 거치고 생성하도록 고쳐주었다.
        - Person.ssn_constructor(ssn_1)
        - Person(year, month, day, sex)
        - 위 두가지 방법으로 instance를 생성할 수 있다.
    
    ```python
    class Person:
        def __init__(self, year, month, day, sex):
            self.year = year
            self.month = month
            self.day = day
            self.sex = sex
    
        def __str__(self):
            return '{}년 {}월 {}일생 {}입니다.'.format(self.year, self.month, self.day, self.sex)
    
        @classmethod # 원래 바깥에 ssn_constructor로 함수를 만들어 class init에 전달해도 된다.
        def ssn_constructor(cls, ssn):
            front, back = ssn.split('-')
            sex = back[0]
    
            if sex == '1' or sex == '2':
                year = '19' + front[:2]
            else:
                year = '20' + front[:2]
    
            if (int(sex) % 2) == 0:
                sex = '여성'
            else:
                sex = '남성'
    
            month = front[2:4]
            day = front[4:6]
    
            return cls(year, month, day, sex)
    
    ssn_1 = '900829-1034356'
    ssn_2 = '051224-4061569'
    
    # 인스턴스를 만들어 주면서 동시에 ssn_constructor를 실행 가능
    # return cls()를 통해서 init을 실행해 준다.
    person_1 = Person.ssn_constructor(ssn_1)
    print(person_1)
    
    person_2 = Person.ssn_constructor(ssn_2)
    print(person_2)
    ```
    

# 3. Static method

**`@staticmethod`** 데코레이터를 사용해서 클래스에 메서드를 선언하여 사용한다. 이 static method는 instance method나 class method와 달리 필수적인 첫번째 매개변수가 할당되지 않는다.(self나 cls가 필요 없다.) 따라서 class와 instance에 독립적이기 때문에 class / instance 속성에 접근하거나 호출하는 것이 불가능하다.

일반적으로 정적 메서드는 유틸리티 메서드를 구현할 때 많이 사용된다.

- Code
    
    ```python
    class StringUtils:
        @staticmethod
        def toCamelcase(text):
            words = iter(text.split("_"))
            return next(words) + "".join(i.title() for i in words)
    
        @staticmethod
        def toSnakecase(text):
            letters = ["_" + i.lower() if i.isupper() else i for i in text]
            return "".join(letters).lstrip("_")
    ```

# 4. Class method vs. Static method

클래스 메서드와 정적 메서드는 별도 인스턴스 생성없이 클래스를 대상으로 클래스 이름 뒤에 바로 `.` 오퍼레이터를 붙여서 호출할 수 있다는 점에서 동일합니다.

차이점은 클래스 메서드를 호출할 때, 첫번째 인자로 클래스 자체가 넘어오기 때문에, 클래스 속성에 접근하거나 다른 클래스 함수를 호출할 수 있습니다. 반면에 정적 메서드를 호출할 때는, 첫번째 인자로 아무것도 넘어오지 않기 때문에, 명시적으로 넘긴 다른 인자만 접근할 수 있습니다.

# 5. Refernece

[Ref. 1](https://schoolofweb.net/blog/posts/%ed%8c%8c%ec%9d%b4%ec%8d%ac-oop-part-4-%ed%81%b4%eb%9e%98%ec%8a%a4-%eb%a9%94%ec%86%8c%eb%93%9c%ec%99%80-%ec%8a%a4%ed%83%9c%ed%8b%b1-%eb%a9%94%ec%86%8c%eb%93%9c-class-method-and-static-method/)
    
[[파이썬] 정적(static) 메서드와 클래스(class) 메서드](https://www.daleseo.com/python-class-methods-vs-static-methods/)