---
title: "Tmux"
excerpt: "본문의 주요 내용을 여기에 입력하세요"

categories:
  - Linux
tags:
  - [tmux, server]

permalink: /Linux/tmux/

toc: true
toc_sticky: true

date: 2022-10-15
last_modified_at: 2022-10-15
---

# tmux

Tmux는 GNU Screen의 대안인 터미널 멀티플렉서입니다. 즉, Tmux 세션을 시작한 다음 해당 세션 내에서 여러 창을 열 수 있습니다. 각 창은 전체 화면을 차지하며 직사각형 창으로 분할할 수 있습니다.

Tmux를 사용하면 한 터미널에 있는 여러 프로그램 간에 쉽게 전환할 수 있으며, 프로그램을 분리한 다음 다른 터미널에 다시 연결할 수 있습니다.

Tmux 세션은 지속적이므로 연결이 끊겨도 Tmux에서 실행 중인 프로그램이 계속 실행됩니다.

Tmux의 모든 명령은 접두사로 시작하며, 기본적으로 Ctrl+b입니다.

# 1. Install

## Linux

```bash
sudo apt install tmux
```

## CentOS

```bash
sudo yum install tmux
```

## MacOS

```bash
brew install tmux
```

# 2. tmux 구성

- session : tmux 실행 단위. 여러개의 window로 구성.
- window : 터미널 화면. 세션 내에서 탭처럼 사용할 수 있음.
- pane : 하나의 window 내에서 화면 분할.
- status bar : 화면 아래 표시되는 상태 막대.

'''  
[ Tree ]  
+--- session  
|    +--- window 1  
|      +--- pane 1  
|      +--- pane 2  
|    +--- window 2  
|      +---pane 1  
'''

# 3. 명령어 정리

## 3.1. base

tmux에서 명령어를 사용하기 위해서는 항상 

```bash
ctrl + b, < key >
```

로 이루어져 있다.

일부 직접 명령어를 입력해야 할 때는 명령어 모드로 진입해야 한다.

```bash
ctrl + b, :
```

## 3.2. commands

### 1) tmux session 시작과 종료

```bash
# tmux 세션 시작
tmux 

# 이름을 붙여서 tmux 세션 시작
tmux new -s {session_name}
tmux new -s {session-name} -n {window-name}
# 현재 열려있는 세션 리스트 보기
tmux ls

# 0: 1 windows (created Thu Jun 30 14:36:05 2022)
# 1: 1 windows (created Thu Jun 30 14:36:25 2022)
# 2: 1 windows (created Thu Jun 30 16:07:53 2022)

# 열려있는 세션 중 하나에 연결하기
tmux a -t {session name}
tmux attach-session -t {session name}

# 세션에서 나오기
ctrl + b, d

# 세션 안에서 세션 끝내기
exit

# 세션 밖에서 세션 끝내기
tmux kill-session -t {session name}
```

### 2) commands in window

- Ctrl+b c : 셸이 있는 새 window을 만듭니다.
- Ctrl+b w : 목록에서 window을 선택합니다.
- Ctrl+b n : 다음 window로 넘어갑니다.
- Ctrl+b 0 : window 0으로 전환합니다(숫자 기준).
- Ctrl+b , : 현재 window 이름 바꾸기
- Ctrl+d : 현재 window 종료

### 3) commands in pane

- Ctrl+b % : 현재 창을 두 개의 pane으로 가로로 분할합니다.
- Ctrl+b " : 현재 창을 두 개의 pane으로 수직으로 분할합니다.
- Ctrl+b o : 다음 pane으로 이동합니다.
- Ctrl+b ; : 현재 pane과 이전 pane 사이를 전환합니다.
- Ctrl+b x : 현재 pane을 닫습니다.

# 4. Plug in

[https://github.com/tmux-plugins/tpm](https://github.com/tmux-plugins/tpm)

- Reference
    
    [Linux : Tmux 설치, 사용하는 방법, 예제, 명령어](https://jjeongil.tistory.com/1361)
    
    [tmux 입문자 시리즈 요약](https://edykim.com/ko/post/tmux-introductory-series-summary/)
    
    [tmux 설치 및 적응기](https://bossm0n5t3r.github.io/posts/74/)
    
    [터미널 화면분할 Tmux 쉽게 사용하기](https://velog.io/@suasue/Ubuntu-%ED%84%B0%EB%AF%B8%EB%84%90-%ED%99%94%EB%A9%B4%EB%B6%84%ED%95%A0-Tmux-%EC%89%BD%EA%B2%8C-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0)