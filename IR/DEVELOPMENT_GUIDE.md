# P2Pro-Viewer 개발 및 환경 설정 가이드

> 본 문서는 `.claude` 대화 기록 및 실제 개발/테스트 경험을 바탕으로 작성되었습니다.  
> 다른 개발자가 새로운 Windows 환경에서 이 프로젝트를 클론하고 바로 실행/수정할 수 있도록 모든 문제 해결 과정과 팁을 담았습니다.

---

## 1. 프로젝트 개요

**InfiRay P2 Pro** (또는 Xinfrared P2 Pro)는 256×192 해상도의 USB UVC 열화상 칩메라입니다.  
공식 InfiRay SDK 없이 Python + OpenCV + pyusb로 역설계(reverse-engineering)하여 제어하는 오픈소스 뷰어입니다.

### 주요 기능 (본 포크에서 추가/개선됨)
- **VTK Structured Grid 낵포기**: 실시간 온도 필드를 VTK 파일로 저장 (ParaView 등으로 시각화 가능)
- **실시간 온도 오버레이**: 화면 중앙 / 최소 / 최대 / 범위 동시 표시
- **Gain 모드 자동 전환**: 시작 시 Low Gain(고온 ~600°C) 자동 설정
- **INFERNO 컬러맵**: MAGMA → INFERNO로 변경
- **Graceful Fallback**: Zadig/libusb 드라이버가 없어도 영상 수신 및 VTK 저장은 정상 동작
- **Windows libusb PATH 자동 설정**: main.py에서 DLL 경로를 자동으로 잡아줌

---

## 2. 하드웨어 요구사항

| 항목 | 사양 |
|------|------|
| 칩메라 | InfiRay P2 Pro (USB-C) |
| 해상도 | 256×192 (열화상) |
| PC 연결 | USB-C 또는 USB-A 케이블 (데이터 전송 지원 케이블 필수) |
| OS | Windows 10/11 (Linux는 udev 규칙 필요) |
| Python | 3.10 이상 |

### Windows에서 장치 인식 확인
장치 관리자 또는 PowerShell로 아래와 같이 뜨면 정상:
```powershell
Get-PnpDevice | Where-Object { $_.FriendlyName -match "USB Camera" }
# 결과: USB Camera  (VID_0BDA&PID_5830)
```
- **VID** = `0x0BDA` (Realtek)
- **PID** = `0x5830`

OpenCV로 스캔하면 보통 `Camera index 1`, 해상도 `256x384`로 잡힙니다.  
(256×192 영상 상단 + 256×192 온도 데이터 하단이 하나의 프레임으로 전송됨)

---

## 3. 드라이버 설치 (핵심)

P2 Pro는 기본적으로 **UVC 웹캠**으로 인식되므로 영상은 바로 볼 수 있습니다.  
하지만 **USB Vendor Command** (Gain 전환, NUC 보정, 셔터 제어 등)를 쓰려면 별도 드라이버가 필요합니다.

### 방법 A: Zadig (권장, 대부분의 경우 동작)

1. [Zadig](https://zadig.akeo.ie/) 다운로드 및 실행
2. `Options` → `List All Devices` 체크
3. 목록에서 **`USB Camera (Interface 0)`** 선택
   - VID가 `0BDA`, PID가 `5830`인지 반드시 확인
4. Driver 칸을 **`libusb-win32`**로 선택
5. **`Install Filter Driver`** (또는 Replace Driver) 클릭

> ⚠️ **주의**: "Replace Driver"는 원래 드라이버를 완전히 덮어쓰므로, 다른 웹캠 소프트웨어에서 P2 Pro를 못 볼 수 있습니다. 문제 발생 시 원래 드라이버로 복구해야 합니다.

### 방법 B: libusb-win32-devel-filter (안전한 필터 방식)

`tools/libusb-win32-devel-filter.exe`를 실행하여 필터 방식으로 설치할 수도 있습니다.  
이 방식은 원래 드라이버를 유지하면서 libusb 접근을 허용합니다.

### 드라이버 설치 후 확인

```bash
python -c "import usb.core; dev = usb.core.find(idVendor=0x0BDA, idProduct=0x5830); print(dev)"
```

`None`이 아닌 장치 정보가 출력되면 성공입니다.

---

## 4. Python 환경 설정

### 필수 패키지 설치

```bash
pip install opencv-python numpy pyusb
```

### libusb DLL 경로 (Windows 필수)

Windows에서는 pyusb가 libusb 백엔드를 찾지 못하는 경우가 많습니다.  
`main.py`에 아래와 같이 자동 설정 코드가 이미 들어가 있지만, 혹시 경로가 다르면 수동으로 수정하세요.

```python
if platform.system() == 'Windows':
    _libusb_dir = os.path.join(
        os.path.expanduser("~"),
        r"AppData\Roaming\Python\Python310\site-packages\libusb\_platform\windows\x86_64"
    )
    if os.path.isdir(_libusb_dir):
        os.environ['PATH'] = _libusb_dir + ';' + os.environ.get('PATH', '')
```

**Python 버전이 다르다면** `Python310` 부분을 `Python311` 등으로 변경하세요.

### 한글 경로 문제

원래 `C:\Users\신지선` 경로에서 실행 시 Python/Kivy 일부 모듈이 한글 경로를 제대로 처리하지 못해 스크립트 실행이 실패했습니다.  
**해결책**: `D:\claude` 등 영문 경로로 프로젝트를 이동하여 사용합니다.

---

## 5. 실행 방법

### 기본 실행

```bash
cd IR
python main.py
```

### 동작 흐름

1. libusb DLL PATH 자동 설정
2. USB command interface 초기화 시도
   - 성공: "USB command interface: OK"
   - 실패: "USB command interface: UNAVAILABLE" → 영상/VTK는 그대로 작동
3. P2 Pro 카메라 자동 검색 (Linux: udev, Windows: 해상도 256×384 매칭)
4. **Low Gain (고온 모드)** 자동 설정 (USB 명령 가능한 경우)
5. 실시간 열화상 영상 표시 시작

---

## 6. 키보드 조작법

| 키 | 기능 |
|----|------|
| **`q`** | 프로그램 종료 |
| **`Enter`** | 현재 프레임을 VTK + PNG로 저장 (`~/P2Pro_VTK/` 또는 `C:\Users\<User>\P2Pro_VTK\`) |
| **`l`** | **Low Gain** — 고온 모드 (~600°C), 화질 낮음 |
| **`h`** | **High Gain** — 일반 모드 (~120°C), 화질 좋음 |
| **`s`** | **NUC 셔터 수동 보정** (Shutter actuate) |
| **`b`** | **배경 보정** (Shutter background) |
| **`d`** | 현재 셔터 상태 읽기 |

> ⚠️ `l`, `h`, `s`, `b`, `d`는 Zadig/libusb 드라이버가 설치되어 있어야 동작합니다.

---

## 7. VTK 파일 구조

`Enter` 키를 누르면 아래와 같은 VTK Structured Grid 파일이 생성됩니다:

```vtk
# vtk DataFile Version 3.0
P2Pro Thermal Data
ASCII
DATASET STRUCTURED_GRID
DIMENSIONS 256 192 1
POINTS 49152 float
0 191 0
1 191 0
...
POINT_DATA 49152
SCALARS Temperature_C float 1
LOOKUP_TABLE default
23.45
23.67
...
```

- **ParaView**에서 `File → Open` 후 그대로 불러오면 됩니다.
- `Temperature_C` 스칼라 필드로 온도 분포를 색상/등고선으로 시각화 가능합니다.

---

## 8. 프로젝트 파일 구조

```
IR/
├── main.py                 # 엔트리 포인트
├── P2Pro/
│   ├── video.py            # 영상 캡처, VTK 저장, 디스플레이, 키 입력 처리
│   ├── P2Pro_cmd.py        # USB vendor commands (Gain, NUC, 색상, 장치정보)
│   ├── recorder.py         # MKV 녹화 (원본 기능)
│   ├── gui.py / gui.kv     # Kivy GUI (원본 기능, 현재 CLI 위주 사용)
│   └── util.py             # 유틸리티
├── tools/
│   └── libusb-win32-devel-filter.exe   # 드라이버 설치 도구
├── 60-p2pro.rules          # Linux udev 규칙
└── README.md / DEVELOPMENT_GUIDE.md
```

---

## 9. 개발 개선 이력 (claude 기록 기반)

| 날짜 | 개선 내용 |
|------|-----------|
| 2026-04-03 | ftobler `infiray_p2_pro_python` 클론 → 기본 OpenCV 뷰어 확인 |
| 2026-04-03 | Pinoerkel `P2Pro-Viewer` 클론 → USB vendor command 기반 고급 제어 시작 |
| 2026-04-03 | `video.py` 수정: VTK Structured Grid 낵포기 함수 추가, INFERNO 컬러맵, 온도 오버레이 |
| 2026-04-03 | `main.py` 수정: Windows libusb PATH 자동 설정, Graceful fallback, 자동 Low Gain |
| 2026-04-03 | `P2Pro_cmd.py` 수정: libusb0 backend 우선 시도, `gain_set_low/high()` 편의 메서드 추가 |
| 2026-04-03 | `SETUP.md` 및 `tools/libusb-win32-devel-filter.exe` 추가 |

### Gain 모드 상세
- **High Gain** (`GAIN_SEL = 1`): -20°C ~ 120°C, 이미지 품질 우수
- **Low Gain** (`GAIN_SEL = 0`): -20°C ~ 550°C (일부 사양 600°C), 이미지 품질 저하
- PBF(분말베드퓨전) 빌드 모니터링 등 고온 측정 시 Low Gain 필수

### 온도 변환 공식
```python
def to_temperature(raw_value):
    return raw_value / 64.0 - 273.15
```

---

## 10. 문제 해결 (Troubleshooting)

### Q1. 장치 관리자에 아예 안 잡힘
- USB-C 케이블이 **데이터 전송 지원** 케이블인지 확인 (충전 전용 케이블이 아님)
- P2 Pro 전원 부족 시 부팅 불가 → 버튼 눌러서 흰색 LED가 켜지는지 확인

### Q2. "USB command interface: UNAVAILABLE"
- Zadig/libusb 드라이버가 설치되지 않은 상태입니다.
- **영상 수신과 VTK 저장은 정상 동작**하며, Gain/NUC 제어만 비활성화됩니다.
- 드라이버 설치 후 재실행하세요.

### Q3. 화면이 뿌옇게 나옴 / 초점이 안 맞음
- P2 Pro는 **고정 초점(Fixed Focus)**입니다. 물리적 초점 조절이 불가능합니다.
- **렌즈 보호 비닐/매크로 렌즈를 제거**하세요.
- 게르마늄 렌즈에 지문이나 먼지가 묻었는지 확인.
- 대상이 20cm 이내면 흐릴 수 있습니다.
- 주기적으로 화면이 멈추거나 흐려지는 것은 **NUC(셔터 보정)**으로, 정상 동작입니다.

### Q4. `cv2.VideoCapture` DSHOW WARN 메시지
```
[ WARN:0@...] global cap.cpp:480 cv::VideoCapture::open VIDEOIO(DSHOW): backend is generally available but can't be used to capture by index
```
- **무시해도 됩니다.** DSHOW가 일부 인덱스를 스킵하면서 출력하는 일반적인 경고입니다.

### Q5. Python 실행 시 `ModuleNotFoundError: No module named 'usb'`
```bash
pip install pyusb
```

### Q6. `libusb` 백엔드를 찾을 수 없음 (`No backend available`)
- libusb DLL이 PATH에 없는 경우입니다.
- `main.py`의 자동 설정 경로를 확인하거나, Zadig을 통해 시스템 전역으로 libusb를 설치하세요.

---

## 11. 참고 자료

- Pinoerkel 원본 저장소: https://github.com/Pinoerkel/P2Pro-Viewer
- ftobler 단순 뷰어: https://github.com/ftobler/infiray_p2_pro_python
- Zadig: https://zadig.akeo.ie/
- EEVblog P2 Pro 리뷰 스레드: https://www.eevblog.com/forum/thermal-imaging/review-infiray-p2-pro-thermal-camera-dongle-for-android-mobile-phones/
- InfiRay 공식 사양: 측정 범위 -20°C ~ 550°C (High Gain: ~120°C, Low Gain: ~550°C)

---

## 12. 향후 개선 아이디어 (원본 로드맵 참고)

- [ ] 의사색상(Palette) 실시간 GUI 변경
- [ ] 방사율(Emissivity) 조정 UI
- [ ] 측정 거리(Distance) 설정 반영
- [ ] MKV 녹화 시 오버레이(온도 눈금, min/max/center) 입힌 영상으로 렌더링
- [ ] CSV 로깅 (시계열 온도 데이터 기록)
- [ ] Android 앱 JPEG 내 radiometry 데이터 파싱

---

*마지막 업데이트: 2026-05-14*  
*작성자: mastajh (claude 기록 기반 정리)*
