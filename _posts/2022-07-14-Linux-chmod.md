---
title: "파일 권한 변경"
excerpt: "파일 권한 변경 방법 설명"

categories:
  - Linux
  - chmod

tags:
  - [chomod]

permalink: /Linux/chmod/

toc: true
toc_sticky: true

date: 2022-07-14
last_modified_at: 2022-07-14
---

# 1. Change mode(chmod)
---
```
chmod [OPTION] [MODE] [FILE]
[ OPTION ]
        -v        : 모든 파일에 대해 모드가 적용되는 진단(diagnostic) 메시지 출력.
        -f        : 에러 메시지 출력하지 않음.
        -c        : 기존 파일 모드가 변경되는 경우만 진단(diagnostic) 메시지 출력.
        -R        : 지정한 모드를 파일과 디렉토리에 대해 재귀적으로(recursively) 적용.
[ MODE ]
        파일에 적용할 모드(mode) 문자열 조합.
          u,g,o,a : 소유자(u), 그룹(g), 그 외 사용자(o), 모든 사용자(a) 지정.
          +,-,=   : 현재 모드에 권한 추가(+), 현재 모드에서 권한 제거(-), 현재 모드로 권한 지정(=)
          r,w,x   : 읽기 권한(r), 쓰기 권한(w), 실행 권한(x)
          X       : "디렉토리" 또는 "실행 권한(x)이 있는 파일"에 실행 권한(x) 적용.
          s       : 실행 시 사용자 또는 그룹 ID 지정(s). "setuid", "setgid".
          t       : 공유모드에서의 제한된 삭제 플래그를 나타내는 sticky(t) bit.
          0~7     : 8진수(octet) 형식 모드 설정
```
# 2. Reference

[리눅스 chmod 명령어 사용법. (Linux chmod command) - 리눅스 파일 권한 변경.](https://recipes4dev.tistory.com/175)