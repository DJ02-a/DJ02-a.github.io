---
title: "저장 장치"
excerpt: "저장 장치의 간단한 소개 및 외장 하드 mount 방법 소개"

categories:
  - Linux
  - storage
  - storage device

tags:
  - [storage device]

permalink: /Linux/storage-device/

toc: true
toc_sticky: true

date: 2022-05-27
last_modified_at: 2022-05-27
---

# 1. 저장 장치
---

- 파티션 : 물리 디스크를 파티션이라는 논리 단위로 나누는 것
- 파일 시스템 : 컴퓨터에서 파일이나 자료를 쉽게 발견 및 접근할 수 있도록 보관 또는 조직하는 체계

> 먼저, 아무 setting도 되지 않은 SSD를 받으면 아무 파티션도 존재하지 않습니다. 하지만 이 SSD에 먼저 파티션 관리 정책 MBR, GPT 중 1개를 골라줍니다. 이후, 해당 SSD를 어떻게 사용할지 파티션을 지정하고 파일 시스템을 설정합니다. SSD는 C드라이브와 D드라이브 2개의 **파티션**으로 나눠졌으며, **파일시스템**은 NTFS입니다.
> 

## 1.1. 파티션

### MBR

- 주 파티션을 4개 까지 생성 가능
- 디스크 용량 최대 2TB까지 인식
- 호환성 : Window 32 비트

### GPT

- 주 파티션을 128개 까지 생성 가능
- 디스크 용량 최대 9.4ZB까지 인식
- 호환성 : windows 32 비트 사용 불가

그냥 GPT 로 하면 된다.

## 1.2. 파일 시스템

- exFAT : 안정성이 떨어지기 때문에 언마운트 잘 하기

<img src="/assets/images/posts_img/2022-05-27-Linux-Drive/1.table.png">

# 2. 저장 공간

---

## 2.1. df 명령어

(disk free)

- df -h(human) : 사람이 보기 쉽게 size 표시

## 2.2. du 명령어

(disk used)

- du -h(human) : 사람이 보기 쉽게 size 표시
- du {folder path} | sort -n : size 크기 별로 정렬하는 방법
(-h를 사용하면 정렬이 이상하게 된다.)    

# 3. HDD Mount

---

[https://csm-kr.tistory.com/9](https://csm-kr.tistory.com/9)

1. 외장하드 꽂기
2. 외장하드 확인
>> sudo fdisk -l   (보통 맨 밑에 있음, 보통 sda1 이지만 아닌 경우 아래 예시 라인들 수정해야함)
3. 외장하드 데이터타입 확인 
>> sudo blkid   (TYPE="ntfs" or "exfat")
4. 외장하드를 마운트할 디렉토리 생성
>> mkdir ./HDD
5. 마운트
5-1. 3에서 확인한 타입이 ntfs 인 경우
>> sudo mount -t ntfs /dev/sda1 ./HDD
5-2. 3에서 확인한 타입이 exfat 인 경우
>> sudo apt-get install exfat-fuse exfat-utils (뭔가 설치해야함)
>> sudo mount -t exfat /dev/sda1 ./HDD
6. 마운트 해제할 경우
>> sudo umount /dev/sda1 ./HDD


# 4. Reference

[리눅스 언마운트(umount) target is busy 발생할 경우](https://boya.tistory.com/174)