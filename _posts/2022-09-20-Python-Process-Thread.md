---
title: "Process & Thread"
excerpt: "Process & Thread"

categories:
  - Python
  - Process
  - Thread
  

tags:
  - [Process]
  - [Thread]

permalink: /Python/Process-Thread/

toc: true
toc_sticky: true

date: 2022-09-20
last_modified_at: 2022-09-20
---

# 1. Process

---

## 1.1. Process 란?

<img src="/assets/images/posts_img/2022-09-20-Python-Process-Thread/1.Operation_System.png">

- **실행중에 있는 프로그램**을 의미한다.
- 스케줄링의 대상이 되는 작업(task)과 같은 의미로 쓰인다.
- **프로세스 내부에는 최소 하나의 스레드(thread)**를 가지고 있는데, 실제로는 스레드(thread)단위로 스케줄링을 한다.
- 하드디스크에 있는 프로그램을 실행하면, 실행을 위해서 **메모리 할당**이 이루어지고, 할당된 메모리 공간으로 바이너리 코드가 올라가게 된다. 이 순간부터 **프로세스**라 불린다.

## 1.2. Process의 구조(PCB)

Process에 대한 정보는 Process Control Block(PCB) 또는 Process descriptor라고 부르는 자료구조에 저장된다. 대부분 `PCB`라고 부른다.

1. PID(Process IDentification)
    - 운영체제가 각 프로세스를 식별하기 위해 부연된 프로세스 식별번호이다.
2. 프로세스 상태
    - CPU는 프로세스를 빠르게 교체하면서 실행하기 때문에 실행중인 프로세스도 있고 대기 중인 프로세스도 있습니다. 그런 프로세스의 상태를 저장한다.
3. 프로세스 카운터
    - CPU가 다음으로 실행할 명령어를 가리키는 값이다. CPU는 기계어를 한 단위씩 읽어서 처리하는데 프로세스를 실행하기 위해 다음으로 실행할 기계어가 저장된 메모리 주소를 가리키는 값이다.
4. 스케줄링 우선순위
    - 운영체제는 여러개의 프로세스를 동시에 실행하는 환경을 제공한다. 운영체제가 여러 개의 프로세스가 CPU에서 실행되는 순서를 결정하는 것을 스케줄링이라고 한다. 이 스케줄링에서 우선순위가 높으면 먼저 실행될 수 있는데 이를 스케줄링 우선순위라고 한다.
5. 권한
    - 프로세스가 접근할 수 있는 자원을 결정하는 정보다. 프로세스마다 어디까지 정보를 접근할 수 있는지에 대한 권한이 필요하다.
6. 프로세스의 부모와 자식 프로세스
    - 최초로 생성되는 init 프로세스를 제외하고 모든 프로세스는 부모 프로세스를 복제해서 생성되고 이 계층관계는 트리를 형성한다. 그래서 각 프로세스는 자식 프로세스와 부모 프로세스에 대한 정보를 가지고 있다.
7. 프로세스의 데이터와 명령어가 있는 메모리 위치를 가리키는 포인터
8. 프로세스에 할당된 자원들을 가리키는 포인터
9. 실행 문맥

## 1.3. Process의 메모리 구조

PCB의 구성 요소 중 **프로세스의 데이터와 명령어가 있는 메모리 위치를 가리키는 포인터**에 대해서 설명한다.

<img src="/assets/images/posts_img/2022-09-20-Python-Process-Thread/2.Memory.png">


- 프로세스 하나당 하나씩 존재한다.
- Code(Text) 영역
    - 프로그램을 실행시키는 실행 파일 내의 명령어들이 올라간다.
    - 소스 코드다
- Data 영역
    - 전역 변수, static 변수의 할당
- Heap 영역
    - 동적 할당을 위한 메모리 영역
- Stack 영역
    - 지역 변수, 함수 호출시 전달되는 인자(파라미터)를 위한 메모리 영역
- Heap 영역과 Stack 영역 사이에 빈 공간
    - 지역변수를 얼마나 사용할지 미리 계산할 수 없기 때문에 지역변수 선언 순서에 따라 스택 영역은 위쪽으로 주소 값을 매기고 동적 할당될 때 힙영역은 아래쪽으로 주소값을 매긴다.

## 1.3. 프로세스 관리

운영체제는 프로세스들의 실행 사이에 프로세스를 교체하고 재시작할 때 오류가 발생하지 않도록 관리해야 한다.

 이를 위해 운영체제는 프로세스의 상태를 **실행**(running), **준비**(ready), **블록**(block) 상태로 분류하고 프로세스를 **상태전이**(state transition)를 통해 체계적으로 관리한다.

<img src="/assets/images/posts_img/2022-09-20-Python-Process-Thread/3.Process_status.png">

### Flow

1. [ 준비 상태 ] 사용자가 프로그램을 실행하면 프로세스가 생성되고 준비리스트에 추가된다.
2. [ 실행 상태 ] 프로세스는 프로세서(CPU)가 사용가능한 상태가 되면 CPU를 할당받는다.
3. [ 블록 상태 ] 프록세스를 다시 사용하기 전에 입출력이 완료대기를 기다려야 하는 상황이라면 완료될때 까지 자신을 블록한다.

**준비** 상태에서 **실행** 상태로 **상태전이** 된다고 한다. 이 과정을 **디스패칭**(dispatching)이라고 한다.

프로세스는 실행 상태에서 CPU를 이용해 연산한 후 CPU를 자발적으로 반납하고 작업이 끝나지 않았으면 다시 준비상태에 들어간다. 운영체제는 다시 준비리스트의 첫번째에 있는 프로세스를 실행상태로 바꾸고 이 과정을 반복한다.

만약 프로세스를 다시 사용하기 전에 입출력이 완료대기를 기다려야 하는 상황이라면 완료될때 까지 자신을 블록한다. 입출력이 완료되면 운영체제가 프로세스를 블록상태에서 준비상태로 다시 전이시킨다.

## 1.4. 프로세스 스케줄링

- CPU는 하나인데 동시에 실행되어야 할 프로세스가 여러개인 경우
    - CPU가 고속으로 여러 프로세스를 일정한 기준으로 순서를 정해서 실행한다.
- 스케줄링(Scheduling)
    - CPU 할당 순서 및 방법을 결정하는 일
    - 기준 : scheduling algorithm을 통해서(우선순위 알고리즘, 라운드 로빈 알고리즘을 혼합함)

- Reference
    



# 2. Thread

---

프로세스는 직렬적으로 한 개의 일을 순서대로 처리하는 루틴을 가지고 있다.  스레드를 사용하면 하나의 프로세스 안에서 여러개의 루틴을 만들어서 병렬적으로 실행할 수 있다. 특히 단순 반복하는 작업을 분리해서 처리할 수 있다.

### Thread 란

- 프로세스 내에서 실행되는 여러 흐름의 단위
- 프로세스의 특정한 수행 경로
- 프로세스가 할당받은 자원을 이용하는 실행의 단위

### 장점

- CPU 사용률 향상
- 효율적인 자원 활용 및 응답성 향상
- 코드 간결 및 유지보수성 향상

## 2.1. Thread 구조

<img src="/assets/images/posts_img/2022-09-20-Python-Process-Thread/4.structure.png">

- 스레드는 프로세스 내에서 각각 Stack만 따로 할당받고 Code, Data, Heap 영역은 공유한다.
- 스레드는 한 프로세스 내에서 동작되는 여러 실행의 흐름으로, 프로세스 내의 주소 공간이나 자원들(힙 등)을 같은 프로세스 내에 스레드끼리 공유하면서 실행된다.
- 프로세스는 다른 프로세스의 메모리에 직접 접근할 수 없다.
- 각각의 스레드는 별도의 레지스터와 스택을 갖고 있지만, 힙 메모리는 서로 읽고 쓸 수 있다.

## 2.2. Thread와 Process 차이

- 프로세스는 각자 프로세스간의 통신에 IPC가 필요하다.
- 각 프로세스는 Code, Data, Heap, Stack 영역을 각자 보유한다.
- 쓰레드는 Code, Data, Heap 영역은 공유하고 Stack영역만 각자 보유한다.
- 프로세스는 생성과 context switching에 많은 비용이 들어간다.
- 쓰레드는 생성과 context switching에 적은 비용이 들어간다.

## 2.3. Example

1. 기본
- Code
    
    ```python
    #Python Thread 예제2
    import logging
    import threading
    
    def get_logger():
        logger = logging.getLogger("Thread Example")
        logger.setLevel(logging.DEBUG)
        #fh = logging.FileHandler("threading.log") #로그 파일 출력
        fh = logging.StreamHandler()
        fmt = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)
    
        logger.addHandler(fh)
        return logger
    
    def execute(number, logger):
        """
        쓰레드에서 실행 할 함수(로깅 사용)
        """
        logger.debug('execute function executing')
        result = number * 2
        logger.debug('execute function ended with: {}'.format(
            result))
    
    if __name__ == '__main__':
        #로거 생성
        logger = get_logger()
        for i, name in enumerate(['Kim', 'Lee', 'Park', 'Cho', 'Hong']):
            my_thread = threading.Thread(
                target=execute, name=name, args=(i,logger))
            my_thread.start()
    ```
    
1. 동기화

쓰레드는 보통 둘 이상의 실행 흐름을 가지고 있기 때문에 공통 메모리 영역의 값을 참조하는 과정에서 동일한 데이터를 조작하는 등의 일련의 과정이 일어나게 된다.

그 과정에서 문제가 발생할 가능성이 있는데 쓰레드의 실행 순서 조정 및 메모리 접근 제한 등으로 문제를 해결하게 되며, 이 때 쓰레드의 동기화 기법이 필요하게 된다.

간단하게 이야기 하면 전역변수를 수정하는 방법을 제시한다.

1. 동기화 - global
- 쓰레드 간에 서로 공유할 수 있는 변수를 하나 만들어 준다.
- Code
    
    ```python
    #Python Thread Synchronization(동기화) 예제1
    
    import threading
    
    tot = 0
    
    def add_total(amount):
        """
        쓰레드에서 실행 할 함수
        전역변수 tot에 amount 더하기
        """
        global tot
        tot += amount
        print (threading.currentThread().getName()+' Not Synchronized  :',tot)
    
    #동기화가 되어 있지 않은 쓰레드 예제
    if __name__ == '__main__':
        for i in range(10000):
            my_thread = threading.Thread(
                target=add_total, args=(1,))
            my_thread.start()
    ```
    

1. 동기화 - Lock
- 여러 쓰레드가 동시간에 전역변수를 편집하게 되면 중복적으로 일을 수행한다.
- global 변수를 다른 쓰레드에서 수정하지 못하도록 막아주는 역할을 한다.
- Code
    
    ```python
    #Python Thread Synchronization(동기화) 예제2
    
    import threading
    
    tot = 0
    lock = threading.Lock() #
    
    def add_total(amount):
        """
        쓰레드에서 실행 할 함수
        전역변수 tot에 amount 더하기
        """
        global tot
        lock.acquire() # 작업이 끝나기 전까지 다른 쓰레드가 공유데이터 접근 금지
        try:
            tot += amount
        finally:
            lock.release() # lock 해제
        print (threading.currentThread().getName()+' Synchronized  :',tot)
    
        """
        또는
    
        global total
        with lock:
            total += amount
        print (threading.currentThread().getName()+' Synchronized  :',tot)
    
        with 문으로 더 간단하게 사용 가능
        """
    
    #동기화가 되어 있는 쓰레드 예제
    if __name__ == '__main__':
        for i in range(10000):
            my_thread = threading.Thread(
                target=add_total, args=(1,))
            my_thread.start()
    ```
    

# 3. Reference


[[운영체제] 프로세스가 뭐지?](https://bowbowbow.tistory.com/16)

[[운영체제] 프로세스란? (스케줄링, 메모리구조, 상태변화)](https://blockdmask.tistory.com/22)

[An introduction to parallel programming using Python's multiprocessing module](https://sebastianraschka.com/Articles/2014_multiprocessing.html)

[multiprocessing Basics - Python Module of the Week](https://pymotw.com/2/multiprocessing/basics.html)

[[Python] Thread and Lock (쓰레드와 락)](https://velog.io/@kho5420/Python-Thread-and-Lock-%EC%93%B0%EB%A0%88%EB%93%9C%EC%99%80-%EB%9D%BD)

[파이썬(Python) - Thread(쓰레드) 설명 및 예제 소스 코드(1) - 기초](https://niceman.tistory.com/138?category=940952)

