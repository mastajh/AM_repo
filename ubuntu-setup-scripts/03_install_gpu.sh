#!/bin/bash
#===============================================================================
# 03_install_gpu.sh - NVIDIA GPU 드라이버 설치 (L40S용)
# Ubuntu 24.04 서버용
#===============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  NVIDIA GPU 드라이버 설치 (Step 3/4)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# GPU 감지
echo -e "${YELLOW}[1] GPU 하드웨어 감지 중...${NC}"
echo "----------------------------------------"

if lspci | grep -i nvidia &>/dev/null; then
    lspci | grep -i nvidia
    echo "----------------------------------------"
    echo -e "${GREEN}✓ NVIDIA GPU 감지됨${NC}"
else
    echo -e "${RED}NVIDIA GPU를 찾을 수 없습니다.${NC}"
    echo "BIOS에서 GPU가 활성화되어 있는지 확인하세요."
    exit 1
fi
echo ""

# 기존 드라이버 확인
echo -e "${YELLOW}[2] 기존 NVIDIA 드라이버 확인...${NC}"
echo "----------------------------------------"

if command -v nvidia-smi &>/dev/null; then
    echo "이미 NVIDIA 드라이버가 설치되어 있습니다:"
    nvidia-smi
    echo "----------------------------------------"
    echo ""
    read -p "기존 드라이버를 유지하시겠습니까? (y=유지/n=재설치): " KEEP_DRIVER
    if [[ "$KEEP_DRIVER" =~ ^[Yy]$ ]]; then
        echo "기존 드라이버를 유지합니다."
        echo ""
        echo "다음 단계: ./04_install_dev_tools.sh"
        exit 0
    fi
else
    echo "NVIDIA 드라이버가 설치되어 있지 않습니다."
fi
echo "----------------------------------------"
echo ""

# 권장 드라이버 확인
echo -e "${YELLOW}[3] 권장 드라이버 확인 중...${NC}"
echo "----------------------------------------"
sudo apt update
ubuntu-drivers devices 2>/dev/null || echo "ubuntu-drivers 정보 없음"
echo "----------------------------------------"
echo ""

# 드라이버 버전 선택
echo -e "${YELLOW}[4] 드라이버 버전 선택${NC}"
echo ""
echo "L40S GPU 권장 드라이버:"
echo "  1) nvidia-driver-550 (권장 - 안정적)"
echo "  2) nvidia-driver-545"
echo "  3) nvidia-driver-535 (LTS)"
echo "  4) 자동 설치 (ubuntu-drivers autoinstall)"
echo ""
read -p "선택하세요 (1-4, 기본값: 1): " DRIVER_CHOICE
DRIVER_CHOICE=${DRIVER_CHOICE:-1}

case $DRIVER_CHOICE in
    1) DRIVER_PACKAGE="nvidia-driver-550" ;;
    2) DRIVER_PACKAGE="nvidia-driver-545" ;;
    3) DRIVER_PACKAGE="nvidia-driver-535" ;;
    4) DRIVER_PACKAGE="auto" ;;
    *) DRIVER_PACKAGE="nvidia-driver-550" ;;
esac

echo ""
echo -e "${YELLOW}[5] 드라이버 설치 중...${NC}"
echo "----------------------------------------"

# Nouveau 드라이버 비활성화
echo "Nouveau 드라이버 비활성화..."
sudo bash -c 'cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
blacklist nouveau
options nouveau modeset=0
EOF'
sudo update-initramfs -u 2>/dev/null || true

# 드라이버 설치
if [ "$DRIVER_PACKAGE" == "auto" ]; then
    sudo ubuntu-drivers autoinstall
else
    sudo apt install -y "$DRIVER_PACKAGE"
fi

echo "----------------------------------------"
echo -e "${GREEN}✓ 드라이버 설치 완료${NC}"
echo ""

# CUDA 툴킷 설치 여부
echo -e "${YELLOW}[6] CUDA 툴킷 설치${NC}"
echo ""
echo "CUDA 툴킷은 딥러닝/ML 개발에 필요합니다."
read -p "CUDA 툴킷을 설치하시겠습니까? (y/n, 기본값: y): " INSTALL_CUDA
INSTALL_CUDA=${INSTALL_CUDA:-y}

if [[ "$INSTALL_CUDA" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}CUDA 툴킷 설치 중...${NC}"
    echo "----------------------------------------"
    sudo apt install -y nvidia-cuda-toolkit
    echo "----------------------------------------"
    echo -e "${GREEN}✓ CUDA 툴킷 설치 완료${NC}"
fi
echo ""

# 설치 완료 안내
echo -e "${RED}========================================${NC}"
echo -e "${RED}  ⚠️  재부팅이 필요합니다!${NC}"
echo -e "${RED}========================================${NC}"
echo ""
echo "GPU 드라이버를 활성화하려면 재부팅이 필요합니다."
echo ""
read -p "지금 재부팅하시겠습니까? (y/n): " REBOOT_NOW

if [[ "$REBOOT_NOW" =~ ^[Yy]$ ]]; then
    echo ""
    echo "5초 후 재부팅합니다..."
    echo "재부팅 후 다음 명령어로 GPU 확인:"
    echo "  nvidia-smi"
    echo ""
    echo "그 다음 단계: ./04_install_dev_tools.sh"
    sleep 5
    sudo reboot
else
    echo ""
    echo "나중에 재부팅하세요: sudo reboot"
    echo ""
    echo "재부팅 후 GPU 확인: nvidia-smi"
    echo "다음 단계: ./04_install_dev_tools.sh"
fi
