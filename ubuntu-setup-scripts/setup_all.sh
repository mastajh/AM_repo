#!/bin/bash
#===============================================================================
# setup_all.sh - 전체 설치 마스터 스크립트
# Ubuntu 24.04 서버 + L40S GPU 개발환경 세팅
#===============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║   Ubuntu 24.04 서버 개발환경 자동 세팅                       ║"
echo "║   (L40S GPU x2 지원)                                         ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

echo -e "${YELLOW}설치 단계:${NC}"
echo "  Step 1: 네트워크 설정"
echo "  Step 2: 기본 패키지 + Node.js + Claude Code"
echo "  Step 3: NVIDIA GPU 드라이버 + CUDA"
echo "  Step 4: 개발 도구 (Docker, Python, Conda)"
echo ""

echo -e "${BLUE}설치 옵션을 선택하세요:${NC}"
echo ""
echo "  1) 전체 설치 (Step 1~4 순차 실행)"
echo "  2) 특정 단계만 실행"
echo "  3) 현재 상태 확인"
echo ""
read -p "선택 (1-3): " MAIN_CHOICE

case $MAIN_CHOICE in
    1)
        echo ""
        echo -e "${YELLOW}전체 설치를 시작합니다...${NC}"
        echo ""

        # Step 1
        bash "$SCRIPT_DIR/01_setup_network.sh"
        if [ $? -ne 0 ]; then
            echo -e "${RED}네트워크 설정 실패. 중단합니다.${NC}"
            exit 1
        fi

        # Step 2
        bash "$SCRIPT_DIR/02_install_base.sh"

        # Step 3 (재부팅 필요하므로 여기서 중단될 수 있음)
        bash "$SCRIPT_DIR/03_install_gpu.sh"

        # Step 4 (재부팅 후 수동 실행)
        bash "$SCRIPT_DIR/04_install_dev_tools.sh"
        ;;

    2)
        echo ""
        echo "실행할 단계를 선택하세요:"
        echo "  1) 네트워크 설정"
        echo "  2) 기본 패키지 + Claude Code"
        echo "  3) GPU 드라이버"
        echo "  4) 개발 도구"
        echo ""
        read -p "선택 (1-4): " STEP_CHOICE

        case $STEP_CHOICE in
            1) bash "$SCRIPT_DIR/01_setup_network.sh" ;;
            2) bash "$SCRIPT_DIR/02_install_base.sh" ;;
            3) bash "$SCRIPT_DIR/03_install_gpu.sh" ;;
            4) bash "$SCRIPT_DIR/04_install_dev_tools.sh" ;;
            *) echo "잘못된 선택입니다." ;;
        esac
        ;;

    3)
        echo ""
        echo -e "${YELLOW}현재 시스템 상태:${NC}"
        echo "========================================"
        echo ""

        # 네트워크
        echo -e "${BLUE}[네트워크]${NC}"
        if ping -c 1 8.8.8.8 &>/dev/null; then
            echo "  ✓ 인터넷 연결됨"
            ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v "127.0.0.1" | while read ip; do
                echo "  IP: $ip"
            done
        else
            echo "  ✗ 인터넷 연결 안됨"
        fi
        echo ""

        # Node.js
        echo -e "${BLUE}[Node.js / Claude Code]${NC}"
        if [ -f "$HOME/.nvm/nvm.sh" ]; then
            source "$HOME/.nvm/nvm.sh" 2>/dev/null
            echo "  Node.js: $(node -v 2>/dev/null || echo '미설치')"
            echo "  Claude Code: $(claude --version 2>/dev/null || echo '미설치')"
        else
            echo "  ✗ NVM 미설치"
        fi
        echo ""

        # GPU
        echo -e "${BLUE}[NVIDIA GPU]${NC}"
        if command -v nvidia-smi &>/dev/null; then
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
                echo "  $line"
            done
        else
            echo "  ✗ NVIDIA 드라이버 미설치"
            if lspci | grep -i nvidia &>/dev/null; then
                echo "  (GPU 하드웨어는 감지됨)"
            fi
        fi
        echo ""

        # Docker
        echo -e "${BLUE}[Docker]${NC}"
        if command -v docker &>/dev/null; then
            echo "  Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"
        else
            echo "  ✗ Docker 미설치"
        fi
        echo ""

        # Python
        echo -e "${BLUE}[Python]${NC}"
        if [ -d "$HOME/.pyenv" ]; then
            export PYENV_ROOT="$HOME/.pyenv"
            export PATH="$PYENV_ROOT/bin:$PATH"
            eval "$(pyenv init -)" 2>/dev/null || true
            echo "  pyenv: 설치됨"
            echo "  Python: $(python --version 2>/dev/null || echo '버전 확인 불가')"
        else
            echo "  ✗ pyenv 미설치"
        fi

        if [ -d "$HOME/miniconda3" ]; then
            echo "  Miniconda: 설치됨"
        fi
        echo ""
        echo "========================================"
        ;;

    *)
        echo "잘못된 선택입니다."
        exit 1
        ;;
esac
