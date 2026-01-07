#!/bin/bash
#===============================================================================
# 01_setup_network.sh - 네트워크 설정 스크립트
# Ubuntu 24.04 서버용
#===============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  네트워크 설정 스크립트 (Step 1/4)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 현재 네트워크 인터페이스 목록 표시
echo -e "${YELLOW}[1] 사용 가능한 네트워크 인터페이스 목록:${NC}"
echo "----------------------------------------"
ip link show | grep -E "^[0-9]+" | awk -F': ' '{print NR") "$2}'
echo "----------------------------------------"
echo ""

# 인터페이스 상세 정보
echo -e "${YELLOW}[2] 인터페이스 상태 상세:${NC}"
echo "----------------------------------------"
ip addr show
echo "----------------------------------------"
echo ""

# 이더넷 인터페이스만 추출 (lo 제외)
INTERFACES=($(ip link show | grep -E "^[0-9]+" | awk -F': ' '{print $2}' | grep -v "lo"))

if [ ${#INTERFACES[@]} -eq 0 ]; then
    echo -e "${RED}이더넷 인터페이스를 찾을 수 없습니다.${NC}"
    exit 1
fi

echo -e "${YELLOW}[3] 감지된 이더넷 인터페이스:${NC}"
for i in "${!INTERFACES[@]}"; do
    echo "  $((i+1))) ${INTERFACES[$i]}"
done
echo ""

# 사용자 선택
read -p "설정할 인터페이스 번호를 선택하세요 (1-${#INTERFACES[@]}): " CHOICE

if [[ ! "$CHOICE" =~ ^[0-9]+$ ]] || [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt ${#INTERFACES[@]} ]; then
    echo -e "${RED}잘못된 선택입니다.${NC}"
    exit 1
fi

SELECTED_IF="${INTERFACES[$((CHOICE-1))]}"
echo ""
echo -e "${GREEN}선택된 인터페이스: ${SELECTED_IF}${NC}"
echo ""

# DHCP 시도
echo -e "${YELLOW}[4] DHCP로 IP 할당 시도 중...${NC}"
sudo dhclient "$SELECTED_IF" 2>/dev/null || true
sleep 3

# 결과 확인
echo ""
echo -e "${YELLOW}[5] 현재 IP 설정:${NC}"
echo "----------------------------------------"
ip addr show "$SELECTED_IF"
echo "----------------------------------------"
echo ""

# 인터넷 연결 테스트
echo -e "${YELLOW}[6] 인터넷 연결 테스트:${NC}"
if ping -c 3 8.8.8.8 &>/dev/null; then
    echo -e "${GREEN}✓ 인터넷 연결 성공!${NC}"

    if ping -c 3 google.com &>/dev/null; then
        echo -e "${GREEN}✓ DNS 확인 성공!${NC}"
    else
        echo -e "${YELLOW}⚠ DNS 문제가 있을 수 있습니다.${NC}"
    fi
else
    echo -e "${RED}✗ 인터넷 연결 실패${NC}"
    echo ""
    echo "수동으로 확인해주세요:"
    echo "  1. 랜선이 제대로 연결되었는지"
    echo "  2. 공유기가 DHCP를 제공하는지"
    echo "  3. 방화벽 설정"
    exit 1
fi

echo ""

# Netplan 영구 설정 여부
read -p "이 설정을 영구 적용하시겠습니까? (y/n): " PERMANENT

if [[ "$PERMANENT" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}[7] Netplan 설정 파일 생성 중...${NC}"

    # 기존 설정 백업
    if [ -f /etc/netplan/00-installer-config.yaml ]; then
        sudo cp /etc/netplan/00-installer-config.yaml /etc/netplan/00-installer-config.yaml.bak
        echo "기존 설정 백업: /etc/netplan/00-installer-config.yaml.bak"
    fi

    # 새 설정 파일 생성
    sudo tee /etc/netplan/01-dhcp-config.yaml > /dev/null << EOF
network:
  version: 2
  renderer: networkd
  ethernets:
    ${SELECTED_IF}:
      dhcp4: true
EOF

    sudo netplan apply
    echo -e "${GREEN}✓ Netplan 설정 완료!${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  네트워크 설정 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "다음 단계: ./02_install_base.sh"
