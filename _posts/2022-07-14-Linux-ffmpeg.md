---
title: "FFMPEG"
excerpt: "FFMPEG 사용방법"

categories:
  - Linux
  - ffmpeg
  - video
  - image

tags:
  - [ffmpeg]

permalink: /Linux/ffmpeg/

toc: true
toc_sticky: true

date: 2022-07-14
last_modified_at: 2022-07-14
---

# 1. FFMPEG

---

Keywords

- bitrate
1초에 해당하는 동영상에 얼마의 비트(bit) 수를 집어 넣느냐를 의미
bitrate가 높을수록 동영상은 더 많은 정보를 가지게 된다. (화질이 높아진다.)

## 1.1 video 2 video

```bash
ffmpeg {input options} -i input.avi {output options} output.avi
```

- -r : 프레임 속도 정하기(-r 24)
    - 비디오를 불러올 때도 옵션을 넣어줘야 한다.
    - 동영상이 원래 60fps라도 불러올 때 아무런 -r 옵션을 붙이지 않으면 30fps로 읽는다.
- -y : 묻지 않고 출력 파일을 덮어쓴다.
- -b(bitrate) : 일반적으로 비디오의 퀄리티를 결정한다.
- -maxrate : bitrate의 최대 수치를 의미
- -ss : 어디에서 시작할지 의미, (단위 : 초)
- -to : 어디까지 변환할지 의미, (단위 : 초)
- -an : 오디오 녹화를 비활성화
- mp4, avi, mov 가 가능하다(더 다양하게 변환 가능)

### fps 조절하기

input 동영상 프레임을 다르게 조절하여 동영상을 만들 수 있다.

```bash
ffmpeg -i input.avi -r 24 output.avi
ffmpeg -r 1 -i input.avi -r 24 output.avi
```

## 1.2. image 2 video

```bash
ffmpeg -f image2 -i pic%04d.png output.mp4
```

- -s : input image의 size 지정(-s 1920x1080)
- -i : 만약 이미지들 이름들이 ‘pic0000.png’,’pic0001.png’라면 위와 같이 쓴다.
- -start_number : 원하는 프레임 number부터 동영상을 만들고 싶은 경우 사용.
- -crf(constant rate factor): 퀄리티를 의미. 낮을수록 좋은 퀄리티를 의미하며 보통 15-25 범위로 사용.
- -c:v : vcodec, 비디오 코덱을 설정하는 옵션. MPEG4를 쓰려면 libxvid를 이용.(libx264)
- -c:a  : acodec, 오디오 코덱을 설정하는 옵션. 주로 acc를 사용한다. mp3를 사용하면 음질이 상대적으로 떨어진다.

## 1.3. video 2 image

```python
ffmpeg -y -i {video_path} %05.png # img%05.png
```

## 1.4. video 2 audio

```bash
ffmpeg -i video.mp4 -vn audio.mp3
```

- -vn : 비디오 녹화를 비활성화

# 2. Reference
[ffmpeg Documentation](https://ffmpeg.org/ffmpeg.html)

[ffmpeg 옵션 정리](https://juyoung-1008.tistory.com/32)

[wiki](https://namu.wiki/w/FFmpeg)
