# P2Pro Thermal Viewer - Setup Guide

InfiRay P2 Pro 열화상 카메라의 실시간 온도 시각화 및 VTK 내보내기 도구.
Pinoerkel/P2Pro-Viewer 포크 기반으로 수정됨.

## 앱 개요

- **실시간 열화상 뷰어**: INFERNO 컬러맵으로 온도 분포 시각화
- **온도 오버레이**: 화면 중앙 온도, Min/Max 표시
- **Gain 전환**: Low Gain(~600도) / High Gain(~120도) 실시간 전환
- **VTK 내보내기**: Enter 키로 현재 프레임을 VTK Structured Grid + PNG 저장
- **NUC 보정**: 셔터 캘리브레이션 수동 트리거

## 실행 방법

```bash
cd C:\Users\신지선\P2Pro-Viewer
python main.py
```

## 단축키

| 키      | 기능                                    |
|---------|-----------------------------------------|
| `Enter` | VTK + PNG 저장 (`~/P2Pro_VTK/`)         |
| `l`     | Low Gain (고온 모드, ~600도)            |
| `h`     | High Gain (일반 모드, ~120도)           |
| `s`     | NUC 셔터 보정                           |
| `b`     | NUC 배경 보정                           |
| `d`     | 셔터 상태 확인                          |
| `q`     | 종료                                    |

## 의존성

### 1. Python 패키지

```bash
pip install opencv-python numpy pyusb keyboard pyaudio ffmpeg-python libusb
```

| 패키지          | 용도                                      |
|-----------------|-------------------------------------------|
| opencv-python   | 카메라 캡처 및 영상 표시                  |
| numpy           | 온도 데이터 배열 처리                     |
| pyusb           | USB vendor command (Gain/NUC 제어)        |
| keyboard        | 핫키 처리 (원본 프로젝트 의존성)          |
| pyaudio         | 오디오 처리 (원본 recorder 모듈 의존성)   |
| ffmpeg-python   | 녹화 기능 (원본 recorder 모듈 의존성)     |
| libusb          | libusb-1.0 DLL 제공 (Windows)             |

### 2. libusb-win32 필터 드라이버 (USB 명령용)

Gain 전환, NUC 보정 등 USB vendor command를 사용하려면 필터 드라이버 필요.
필터 드라이버 없이도 영상 스트림 + VTK 저장은 동작함.

**설치 방법 (권장 - libusb-win32 필터 설치 도구):**

1. `tools\libusb-win32-devel-filter.exe` 실행 (프로젝트에 포함됨)
2. 장치 목록에서 **USB Camera** (VID=0BDA, PID=5830) 선택
3. Install 클릭

> 주의: Zadig 등으로 "Replace Driver"를 하면 UVC 영상 스트림이 깨짐.
> 깨진 경우: 장치 관리자 > USB Camera 우클릭 > 디바이스 제거 > USB 재연결로 복구.

### 포함된 도구 (`tools/` 폴더)

| 파일                              | 용도                                  |
|-----------------------------------|---------------------------------------|
| `libusb-win32-devel-filter.exe`   | libusb-win32 필터 드라이버 설치       |

### 3. 카메라 인덱스 확인

P2 Pro가 카메라 인덱스 1이 아닌 다른 번호에 잡힐 수 있음.
`main.py`의 `vid.open(cam_cmd, 1,)` 에서 숫자를 변경.

확인 방법:
```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Camera {i}: {w}x{h}')
        cap.release()
```
P2 Pro는 **256x384** 해상도로 잡힘.

## VTK 출력 형식

- 파일 위치: `C:\Users\신지선\P2Pro_VTK\thermal_YYYYMMDD_HHMMSS.vtk`
- 포맷: VTK Structured Grid (ASCII)
- 격자: 256 x 192 (카메라 해상도)
- 데이터: `Temperature_C` (float) - 각 포인트의 섭씨 온도
- ParaView, VTK 라이브러리 등에서 열 수 있음

## 온도 변환 공식

```
temperature_celsius = raw_16bit_value / 64.0 - 273.15
```

## 원본 프로젝트

- https://github.com/Pinoerkel/P2Pro-Viewer (이 저장소의 포크 원본)
- https://github.com/LeoDJ/P2Pro-Viewer (최초 원본)

## 수정 사항 (원본 대비)

1. `P2Pro_cmd.py`: libusb0 백엔드 우선 사용 (libusb-win32 필터 드라이버 호환)
2. `video.py`: 온도 오버레이, VTK 저장, cam_cmd=None 허용, FPS 체크 완화
3. `main.py`: libusb DLL PATH 자동 설정, Low Gain 자동 시작, 카메라 ID 지정
