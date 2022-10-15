---
title: "Multiprocessing"
excerpt: "Multiprocessing & Parmap"

categories:
  - Python
  - Multiprocessing
  - Parmap
  - Process

tags:
  - [Multiprocessing]
  - [Parmap]
  - [Process]

permalink: /Python/Multiprocessing/

toc: true
toc_sticky: true

date: 2022-09-12
last_modified_at: 2022-09-12
---

# 1. Multiprocessing

병렬처리 프로그래밍(parallel programming)에는 두가지 접근법이 있다. Thread 또는 multiple processes를 이용한 방법이다.

여러 thread에 어떤 일을 시킨다면  single process에 ‘sut-tasks’로 

## 1.1. process

파이썬 코드를 작성한 후 이를 실행시키면 파이썬 인터프리터가 코드를 해석한 후 실행한다.

프로세스 관점에서 보면 이를 메인 프로세스(Main process)라고 부를 수 있다.

```python
import multiprocessing as mp

if __name__ == "__main__":
    proc = mp.current_process()
    print(proc.name)
    print(proc.pid)

# MainProcess
# 13248
```

`mp.current_process` 를 호출하면 현재 실행되는 프로세스에 대한 정보를 담고 있는 ‘객체'를 얻을 수 있다.

PID란 운영체제가 각 프로세스에게 부여한 고유 번호로서 프로세스의 우선 순위를 조정하거나 종료하는 등 다양한 용도로 사용된다.

```python
ps -ef
ps gv
```

위 명령어로 현재 cpu에서 진행되고 있는 프로세스를 볼 수 있다.

## 1.2. 프로세스 스포닝(spawning)

부모 프로세스(Parent Process)가 운영체제에 요청하여 자식 프로세스(Child Process)를 새로 만들어 내는 과정을 스포닝이라 부른다.

일반적으로 부모 프로레스가 처리할 작업이 많은 경우 자식 프로세스를 새로 만들어 일부 작업을 자식 프로세스에게 위임하여 처리한다.

<img src="/assets/images/posts_img/2022-09-12-Python-Multiprocessing/1.graph.png">

기본적인 사용 방법은 multiprocessing 모듈에서 Process 클래스의 인스턴스를 생성한 후 start() 메소드를 호출하면 된다. Process 클래스의 인스턴스를 생성할 때 생성될 자신 프로세스의 이름과 할 일(함수 명)을 전달한다.

```python
import multiprocessing as mp

def worker():
    print("SubProcess End")

if __name__ == "__main__":
    # process spawning
    p = mp.Process(name="SubProcess", target=worker)
    p.start()
```

- Reference
    


## 1.3. Pool

- Reference
    
    [An introduction to parallel programming using Python's multiprocessing module](https://sebastianraschka.com/Articles/2014_multiprocessing.html)
    
    [파이썬(Python) - multiprocessing(멀티프로세싱) 설명 및 예제(1) - Pool](https://niceman.tistory.com/145)
    

# 2. Parmap

---

위에 있는 방법으로 multiprocessing 으로 코드를 작성하면 번거롭고 귀찮다. thread를 제어하기 위해서 여러 줄을 작성해야 한다. Pool을 올바르게 닫아주지 않으면 잔여 프로세스가 계속 유지되어 유지보수가 어렵다.

다른 문제 : 

[많은 좀비 프로세스를 이끄는 파이썬 멀티프로세싱](https://sdr1982.tistory.com/282)

parmap package를 사용하면 한줄로 multiprocessing을 진행할 수 있다.

# 3. Reference

[https://github.com/zeehio/parmap](https://github.com/zeehio/parmap)

[Python 강좌 : 제 46강 - 프로세스 기반 병렬 처리](https://076923.github.io/posts/Python-46/)
    