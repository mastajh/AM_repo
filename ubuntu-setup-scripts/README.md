# Ubuntu 24.04 서버 개발환경 세팅 스크립트

블레이드 서버 + L40S GPU x2 환경을 위한 자동 설치 스크립트

## 📁 파일 구조

```
ubuntu-setup-scripts/
├── setup_all.sh          # 메인 스크립트 (전체/선택 설치)
├── 01_setup_network.sh   # 네트워크 설정
├── 02_install_base.sh    # 기본 패키지 + Node.js + Claude Code
├── 03_install_gpu.sh     # NVIDIA 드라이버 + CUDA
├── 04_install_dev_tools.sh # Docker, Python, Miniconda
└── README.md
```

## 🚀 빠른 시작

### 방법 1: USB로 스크립트 복사 후 실행

```bash
# USB 마운트
sudo mount /dev/sdb1 /mnt

# 스크립트 복사
cp -r /mnt/ubuntu-setup-scripts ~/

# 실행 권한 부여
chmod +x ~/ubuntu-setup-scripts/*.sh

# 메인 스크립트 실행
cd ~/ubuntu-setup-scripts
./setup_all.sh
```

### 방법 2: 단계별 수동 실행

```bash
# Step 1: 네트워크 (랜선 연결 후)
./01_setup_network.sh

# Step 2: 기본 패키지 + Claude Code
./02_install_base.sh

# Step 3: GPU 드라이버 (재부팅 필요!)
./03_install_gpu.sh

# [재부팅 후]

# Step 4: 개발 도구
./04_install_dev_tools.sh
```

## 📋 설치 내용

| 단계 | 내용 | 비고 |
|------|------|------|
| Step 1 | 네트워크 설정 | DHCP 자동, Netplan 영구 설정 |
| Step 2 | Node.js + Claude Code | NVM 사용, SSH 서버 포함 |
| Step 3 | NVIDIA Driver 550 + CUDA | L40S 최적화, 재부팅 필요 |
| Step 4 | Docker + pyenv + Miniconda | NVIDIA Container Toolkit 포함 |

## 💡 유용한 명령어

```bash
# GPU 상태 확인
nvidia-smi

# Claude Code 실행
claude

# Docker에서 GPU 테스트
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Python 버전 변경 (pyenv)
pyenv install 3.12.0
pyenv global 3.12.0
```

## ⚠️ 주의사항

1. **재부팅**: GPU 드라이버 설치 후 반드시 재부팅 필요
2. **Docker 권한**: 재로그인 후 sudo 없이 docker 사용 가능
3. **bashrc 적용**: 각 단계 완료 후 `source ~/.bashrc` 실행

## 🔧 문제 해결

### 네트워크 안됨
```bash
# 인터페이스 확인
ip link show

# 수동 DHCP
sudo dhclient <인터페이스명>
```

### GPU 안 잡힘
```bash
# 하드웨어 확인
lspci | grep -i nvidia

# 드라이버 재설치
sudo apt install --reinstall nvidia-driver-550
sudo reboot
```

### Claude Code 인증 (Headless 서버)
```bash
claude --headless
# 표시되는 URL을 다른 컴퓨터에서 열어 인증
```
