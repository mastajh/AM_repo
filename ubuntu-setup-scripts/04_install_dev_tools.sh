#!/bin/bash
#===============================================================================
# 04_install_dev_tools.sh - 개발 도구 설치 (Docker, Python, etc.)
# Ubuntu 24.04 서버용
#===============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  개발 도구 설치 스크립트 (Step 4/4)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# GPU 상태 확인
echo -e "${YELLOW}[1] GPU 상태 확인...${NC}"
echo "----------------------------------------"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo -e "${GREEN}✓ GPU 드라이버 정상 작동${NC}"
else
    echo -e "${YELLOW}⚠ nvidia-smi를 찾을 수 없습니다.${NC}"
    echo "  GPU 드라이버가 설치되지 않았거나 재부팅이 필요합니다."
fi
echo "----------------------------------------"
echo ""

# 설치할 도구 선택
echo -e "${YELLOW}[2] 설치할 개발 도구를 선택하세요${NC}"
echo ""
echo "  1) Docker + NVIDIA Container Toolkit"
echo "  2) Python (pyenv + 최신 Python)"
echo "  3) Miniconda (Anaconda 경량 버전)"
echo "  4) 모두 설치"
echo "  5) 선택 설치 (개별 선택)"
echo ""
read -p "선택하세요 (1-5, 기본값: 4): " INSTALL_CHOICE
INSTALL_CHOICE=${INSTALL_CHOICE:-4}

INSTALL_DOCKER=false
INSTALL_PYENV=false
INSTALL_CONDA=false

case $INSTALL_CHOICE in
    1) INSTALL_DOCKER=true ;;
    2) INSTALL_PYENV=true ;;
    3) INSTALL_CONDA=true ;;
    4) INSTALL_DOCKER=true; INSTALL_PYENV=true; INSTALL_CONDA=true ;;
    5)
        read -p "Docker 설치? (y/n): " ans; [[ "$ans" =~ ^[Yy]$ ]] && INSTALL_DOCKER=true
        read -p "pyenv 설치? (y/n): " ans; [[ "$ans" =~ ^[Yy]$ ]] && INSTALL_PYENV=true
        read -p "Miniconda 설치? (y/n): " ans; [[ "$ans" =~ ^[Yy]$ ]] && INSTALL_CONDA=true
        ;;
esac

echo ""

#===============================================================================
# Docker 설치
#===============================================================================
if $INSTALL_DOCKER; then
    echo -e "${YELLOW}[3] Docker 설치 중...${NC}"
    echo "----------------------------------------"

    # 기존 Docker 확인
    if command -v docker &>/dev/null; then
        echo "Docker가 이미 설치되어 있습니다: $(docker --version)"
    else
        # Docker 공식 설치 스크립트 사용
        curl -fsSL https://get.docker.com | sudo sh

        # 현재 사용자를 docker 그룹에 추가
        sudo usermod -aG docker $USER
    fi

    echo -e "${GREEN}✓ Docker 설치 완료${NC}"
    echo ""

    # NVIDIA Container Toolkit 설치
    echo -e "${YELLOW}NVIDIA Container Toolkit 설치 중...${NC}"

    # GPG 키 추가
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true

    # 저장소 추가
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt update
    sudo apt install -y nvidia-container-toolkit

    # Docker 데몬 설정
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    echo -e "${GREEN}✓ NVIDIA Container Toolkit 설치 완료${NC}"
    echo "----------------------------------------"
    echo ""
fi

#===============================================================================
# pyenv 설치
#===============================================================================
if $INSTALL_PYENV; then
    echo -e "${YELLOW}[4] pyenv 설치 중...${NC}"
    echo "----------------------------------------"

    # 의존성 설치
    sudo apt install -y make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
        libffi-dev liblzma-dev

    # pyenv 설치
    if [ -d "$HOME/.pyenv" ]; then
        echo "pyenv가 이미 설치되어 있습니다."
    else
        curl https://pyenv.run | bash
    fi

    # 환경 변수 설정
    if ! grep -q 'PYENV_ROOT' ~/.bashrc; then
        cat >> ~/.bashrc << 'EOF'

# pyenv 설정
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
EOF
    fi

    # pyenv 로드
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)" 2>/dev/null || true

    echo ""
    echo "Python 버전 설치 (권장: 3.11 또는 3.12)"
    read -p "설치할 Python 버전 (예: 3.11.9, 기본값: 3.11.9): " PY_VERSION
    PY_VERSION=${PY_VERSION:-3.11.9}

    echo ""
    echo "Python $PY_VERSION 설치 중... (시간이 걸릴 수 있습니다)"
    pyenv install "$PY_VERSION" || true
    pyenv global "$PY_VERSION"

    echo -e "${GREEN}✓ pyenv + Python $PY_VERSION 설치 완료${NC}"
    echo "----------------------------------------"
    echo ""
fi

#===============================================================================
# Miniconda 설치
#===============================================================================
if $INSTALL_CONDA; then
    echo -e "${YELLOW}[5] Miniconda 설치 중...${NC}"
    echo "----------------------------------------"

    if [ -d "$HOME/miniconda3" ]; then
        echo "Miniconda가 이미 설치되어 있습니다."
    else
        # Miniconda 다운로드 및 설치
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p $HOME/miniconda3
        rm /tmp/miniconda.sh

        # conda 초기화
        $HOME/miniconda3/bin/conda init bash
    fi

    echo -e "${GREEN}✓ Miniconda 설치 완료${NC}"
    echo "----------------------------------------"
    echo ""
fi

#===============================================================================
# 최종 확인
#===============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  설치 완료 요약${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo -e "${YELLOW}설치된 도구:${NC}"
echo "----------------------------------------"

# Node.js / Claude Code
if command -v node &>/dev/null 2>&1 || [ -f "$HOME/.nvm/nvm.sh" ]; then
    source "$HOME/.nvm/nvm.sh" 2>/dev/null || true
    echo "✓ Node.js: $(node -v 2>/dev/null || echo '설치됨')"
    echo "✓ Claude Code: 설치됨"
fi

# Docker
if command -v docker &>/dev/null; then
    echo "✓ Docker: $(docker --version 2>/dev/null | cut -d' ' -f3 | tr -d ',')"
fi

# NVIDIA
if command -v nvidia-smi &>/dev/null; then
    echo "✓ NVIDIA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
fi

# pyenv
if [ -d "$HOME/.pyenv" ]; then
    echo "✓ pyenv: 설치됨"
fi

# Miniconda
if [ -d "$HOME/miniconda3" ]; then
    echo "✓ Miniconda: 설치됨"
fi

echo "----------------------------------------"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ⚠️  중요: 설정 적용${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "새 터미널을 열거나 다음 명령어 실행:"
echo ""
echo -e "  ${GREEN}source ~/.bashrc${NC}"
echo ""

if $INSTALL_DOCKER; then
    echo "Docker를 sudo 없이 사용하려면 로그아웃 후 재로그인하세요."
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  🎉 모든 설치가 완료되었습니다!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Claude Code 시작: claude"
echo ""
