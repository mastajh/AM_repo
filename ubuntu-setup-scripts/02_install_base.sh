#!/bin/bash
#===============================================================================
# 02_install_base.sh - 기본 패키지 + Node.js + Claude Code 설치
# Ubuntu 24.04 서버용
#===============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  기본 패키지 설치 스크립트 (Step 2/4)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 인터넷 연결 확인
echo -e "${YELLOW}[1] 인터넷 연결 확인...${NC}"
if ! ping -c 1 8.8.8.8 &>/dev/null; then
    echo -e "${RED}인터넷 연결이 안됩니다. 01_setup_network.sh를 먼저 실행하세요.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 인터넷 연결 확인됨${NC}"
echo ""

# 시스템 업데이트
echo -e "${YELLOW}[2] 시스템 패키지 업데이트 중...${NC}"
echo "----------------------------------------"
sudo apt update
sudo apt upgrade -y
echo "----------------------------------------"
echo -e "${GREEN}✓ 시스템 업데이트 완료${NC}"
echo ""

# 필수 패키지 설치
echo -e "${YELLOW}[3] 필수 패키지 설치 중...${NC}"
echo "----------------------------------------"
sudo apt install -y \
    curl \
    wget \
    git \
    build-essential \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    htop \
    vim \
    unzip
echo "----------------------------------------"
echo -e "${GREEN}✓ 필수 패키지 설치 완료${NC}"
echo ""

# SSH 서버 설치 (원격 접속용)
echo -e "${YELLOW}[4] SSH 서버 설치...${NC}"
if ! command -v sshd &>/dev/null; then
    sudo apt install -y openssh-server
    sudo systemctl enable ssh
    sudo systemctl start ssh
fi
echo -e "${GREEN}✓ SSH 서버 활성화됨${NC}"
echo ""

# 현재 IP 주소 표시
echo -e "${BLUE}원격 접속 주소:${NC}"
ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v "127.0.0.1" | while read ip; do
    echo "  ssh $(whoami)@$ip"
done
echo ""

# NVM + Node.js 설치
echo -e "${YELLOW}[5] NVM (Node Version Manager) 설치 중...${NC}"
echo "----------------------------------------"

# 이미 설치되어 있는지 확인
if [ -d "$HOME/.nvm" ]; then
    echo "NVM이 이미 설치되어 있습니다."
else
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
fi

# NVM 로드
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

echo "----------------------------------------"
echo -e "${GREEN}✓ NVM 설치 완료${NC}"
echo ""

# Node.js LTS 설치
echo -e "${YELLOW}[6] Node.js LTS 설치 중...${NC}"
echo "----------------------------------------"
nvm install --lts
nvm use --lts
nvm alias default 'lts/*'

echo ""
echo "Node.js 버전: $(node -v)"
echo "npm 버전: $(npm -v)"
echo "----------------------------------------"
echo -e "${GREEN}✓ Node.js 설치 완료${NC}"
echo ""

# Claude Code 설치
echo -e "${YELLOW}[7] Claude Code 설치 중...${NC}"
echo "----------------------------------------"
npm install -g @anthropic-ai/claude-code
echo "----------------------------------------"
echo -e "${GREEN}✓ Claude Code 설치 완료${NC}"
echo ""

# 설치 확인
echo -e "${YELLOW}[8] 설치 확인:${NC}"
echo "----------------------------------------"
echo "Node.js: $(node -v)"
echo "npm: $(npm -v)"
echo "Claude Code: $(claude --version 2>/dev/null || echo '설치됨')"
echo "----------------------------------------"
echo ""

# bashrc 업데이트 안내
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  중요: 새 터미널에서 사용하려면${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "다음 명령어를 실행하거나 새 터미널을 여세요:"
echo ""
echo -e "  ${GREEN}source ~/.bashrc${NC}"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  기본 패키지 설치 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Claude Code 실행 방법:"
echo "  1. source ~/.bashrc"
echo "  2. claude"
echo ""
echo "다음 단계: ./03_install_gpu.sh (GPU 드라이버 설치)"
