#!/usr/bin/env python3
"""
적층제조 공정 분석 애플리케이션 (Enhanced Streamlit 버전)
- 새로운 LLM_Ready_Report.txt 포맷 (PROCESS_HEALTH 섹션 포함) 연동
- 통계 데이터를 장인의 암묵지처럼 해석하여 AM 전문가 용어로 변환
- 재현성 있는 포맷과 그래프 삽입 지원
"""

import os
import re
import streamlit as st
from datetime import datetime
from pathlib import Path
import time
from typing import List, Optional, Any, Dict, TYPE_CHECKING
import json

# 타입 체킹을 위한 조건부 임포트
if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType

# Pillow 라이브러리 임포트 시도
try:
    from PIL import Image
    from PIL.Image import Image as PILImage
except ImportError:
    Image = None
    PILImage = Any  # type: ignore

# google-generativeai 라이브러리 임포트 시도
try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore

# 페이지 설정
st.set_page_config(
    page_title="AM 공정 분석 도구 v3.0",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS - 건강 상태별 색상 및 모던 UI
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .health-healthy {
        background: linear-gradient(135deg, #4CAF50 0%, #81C784 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
    }
    .health-moderate {
        background: linear-gradient(135deg, #FF9800 0%, #FFB74D 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
    }
    .health-high-risk {
        background: linear-gradient(135deg, #f44336 0%, #e57373 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    h1 { color: #2c3e50; font-weight: 700; }
    h2 { color: #34495e; font-weight: 600; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.5rem; }
    h3 { color: #7f8c8d; font-weight: 500; }
    .report-section {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# 모델 매핑 딕셔너리
MODEL_MAPPING = {
    "간단 (Flash-Lite)": "gemini-2.5-flash-lite",
    "보통 (Flash)": "gemini-2.5-flash",
    "고급 (Pro)": "gemini-2.5-pro"
}

# ============================================================
# PROCESS_HEALTH 연동 프롬프트 (새 포맷)
# ============================================================

AM_EXPERT_PROMPT = """<ROLE>
역할: L-PBF/EBM/DED 등 적층제조(AM) 공정 품질진단 전문가
대상: 통계 비전문가인 현장 엔지니어 및 공정 관리자
핵심 원칙: **통계 데이터를 장인의 암묵지처럼 해석**하여 AM 전문가가 직관적으로 판단할 수 있게 변환

문체 규칙:
- 개조식 단문, 명사형 종결
- 통계 용어 최소화, AM 터미놀러지 우선 사용
- 수치: 소수점 1~2자리, 단위 필수 (분, %, 1/s, Hz, ppm)
</ROLE>

<AM_TERMINOLOGY>
필수 사용 용어 (통계 → AM 변환):
- SVD Mode → 공정 지배 패턴 (예: "Mode1이 98% = 단일 패턴이 공정 지배")
- ICA Component → 독립 신호 분리 결과 (예: "IC 8개 모두 impulsive = 불안정 신호 다수")
- DMD Growth Rate → 시간 성장률 (예: "양의 성장률 = 점진적 악화 징후")
- Energy Concentration → 에너지 집중도 (예: "Mode1 >80% = 안정적 단일 모드 지배")
- CV (Coefficient of Variation) → 변동계수 (예: "CV >10% = 센서 출렁임 주의")
- Anomaly Cluster → 이상 구간 (연속적 이상 발생 지점)

AM 현장 용어:
- 리코터(Recoater): 분말 도포 장치
- 해치(Hatch): 내부 스캔 패턴
- 콘투어(Contour): 외곽 스캔 패턴
- 가스 퍼지(Gas Purge): 챔버 가스 순환
- O₂ ppm: 산소 농도 (낮을수록 양호, 보통 <500ppm 목표)
- 스패터(Spatter): 용융풀에서 튀는 분말/금속
- 키홀(Keyhole): 과도 에너지로 인한 깊은 용융풀
- LOF (Lack of Fusion): 불완전 용융 결함
- 에너지 밀도: 레이저 출력/스캔속도/해치간격의 함수
- 열 누적: 빌드 진행 중 열 축적 현상
</AM_TERMINOLOGY>

<PROCESS_HEALTH_INTERPRETATION>
**PROCESS_HEALTH 섹션 해석 가이드:**

1. overall_status 해석:
   - HEALTHY (health_score ≥0.85): 공정 정상. 모니터링 유지.
   - MODERATE_RISK (0.60~0.85): 주의 필요. 예방적 점검 권장.
   - HIGH_RISK (<0.60): 즉시 조치 필요. 심각한 이상 징후.

2. energy_concentration_status 해석:
   - STABLE: Mode1 에너지 >80%. 단일 패턴 지배. 예측 가능한 공정.
   - WARNING: Mode1 에너지 50~80%. 복합 패턴. 모니터링 강화.
   - UNSTABLE: Mode1 에너지 <50%. 다중 패턴 혼재. 공정 불안정.

3. category_balance_status 해석:
   - BALANCED: motion/gas 비율 0.5~2.0. 센서 카테고리 균형.
   - MOTION_DOMINANT: 스캔 시스템(갈보/서보) 이상 징후.
   - GAS_DOMINANT: 가스/분위기 시스템 이상 징후.

4. critical_issues / warnings 해석:
   - ICA problematic ratio >50%: 독립 신호 대부분이 비정상 → 심각
   - Oxygen sensors dominating: 산소 센서가 공정 지배 → 분위기 문제
   - High CV: 해당 센서 출렁임 심함 → 캘리브레이션/점검 필요
</PROCESS_HEALTH_INTERPRETATION>

<SAFETY_GUARDS>
- 원인 단정 금지. "의심", "가능성", "징후" 표현 사용.
- 대안 가설 1개 병기. Confidence(High/Med/Low) 표기.
- 데이터 부족/불일치 시 '판단 보류' 또는 '추가 확인 필요' 명시.
- 그래프는 번호(그래프 N)로만 인용. 본문 상세 묘사는 부록에서만.
- 동일 수치 재인용 금지. 최초 표만 제시, 이후 'KPI 표 참조'.
- 현장 안전 우선: 액션은 가역적·저비용·위험저감 순으로 제시.
</SAFETY_GUARDS>

<CONSISTENCY_RULES>
Self-consistency(텍스트 vs 플롯) 필수:
- 각 항목을 MATCH/MISMATCH로 표기
- MISMATCH 발생 시: 결론 강도 1단계 하향, 원인 1문장 기재

Confidence 산정:
- High: 증거 ≥2종 일치 + Self-consistency 대부분 MATCH
- Med: 증거 ≥1종 일치 또는 일부 불확실
- Low: 증거 부족 또는 MISMATCH 존재
</CONSISTENCY_RULES>

<DECISION_RULES>
위험 신호등 기준 (PROCESS_HEALTH 기반):
- 🔴 HIGH_RISK: health_score <0.60 또는 critical_issues 존재 또는 다운타임 >30분
- 🟡 MODERATE_RISK: health_score 0.60~0.85 또는 warnings 존재
- 🟢 HEALTHY: health_score ≥0.85 AND warnings 최소

결론 강도 억제:
- 단일 지표로 중대한 결론 금지
- 서로 다른 출처 2개 이상 합의 필요 (SVD/ICA/DMD/IForest)
</DECISION_RULES>

<REPORT_STRUCTURE>
## 1. 서론
- 공정: {process_type}. 장비/소재: {machine}/{material}.
- 목적: 빌드 안정성 점검 및 이상 원인 가설 도출.
- 데이터: 원본 {shape_original}, 처리 {shape_processed}, 해상도 {dt_sec}s.
- **공정 건강 상태: {overall_status} (점수: {health_score}/1.00)**
- 범위: 통계 신호 기반. 장비 이벤트 로그/현장 점검 미포함.

## 2. 핵심 지표(KPI) 요약 ※표 형식 고정
| 항목 | 값 | 단위 | AM 해석 |
|---|---:|:---:|---|
| 공정 건강 점수 | {health_score} | /1.00 | 신호등 {risk_emoji} |
| 에너지 집중도 | {mode1_energy_pct} | % | {energy_status} |
| SVD 유효 모드 | {significant_modes} | 개 | 공정 복잡도 |
| 90% 에너지 컴포넌트 | {energy_90_components} | 개 | 지배 패턴 수 |
| ICA 문제 비율 | {ica_problematic_ratio} | % | 독립신호 이상률 |
| DMD 불안정 모드 | {total_unstable_modes} | 개 | 성장 신호 존재 |
| DMD 최대 성장률 | {max_growth_rate} | 1/s | 열 누적/진동 추정 |
| 이상률(SVD) | {svd_anomaly_rate} | % | 선형 이상 비율 |
| 이상치(IForest) | {anomaly_count} | 개 | 비선형 이상 지점 |
- 요약 판단: {summary_judgment}
- 주요 원인 가설 + 대안 가설. Confidence={conf}.
- 즉시 조치 방향 1문장.

## 3. 공정 상태 해석 (AM 관점) ※그래프는 번호만 인용
### 3.1 가스·분위기 (O₂ ppm, 가스 퍼지, 필터 ΔP)
- 증거: {gas_evidence}. (그래프 {gas_graphs} 참조)
- 해석: 보호가스 유지/스패터 제거 적정성.
- 영향: 산화/LOF·기공 위험도. Confidence={gas_conf}.

### 3.2 레이저·스캔 (파워, 해치, 콘투어)
- 증거: {laser_evidence}. (그래프 {laser_graphs} 참조)
- 해석: 에너지 밀도/키홀·스패터 위험도.
- 영향: 용융풀 안정/표면 조도. Confidence={laser_conf}.

### 3.3 열·스테이지 (열 누적, 플랫폼, 리코터)
- 증거: {thermal_evidence}. (그래프 {thermal_graphs} 참조)
- 해석: 저주파 성장→열 누적 또는 리코터 간섭.
- 영향: 변형/워핑/리코터 충돌 리스크. Confidence={thermal_conf}.

## 4. 위험도 평가 (신호등) ※표 형식 고정
| 순위 | 위험 요인(가설) | 영향도 | 근거 | 조치 우선 | Confidence |
|---:|---|:---:|---|---|---|
| 1 | {risk1} | {emoji1} | {evidence1} | 즉시 | {conf1} |
| 2 | {risk2} | {emoji2} | {evidence2} | 1~2주 | {conf2} |
| 3 | {risk3} | {emoji3} | {evidence3} | 정기 | {conf3} |

### 4.1 문제 센서 (상위)
| 센서 | 이상 유형 | 정량 근거 | 권장 조치 |
|---|---|---|---|
| {sensor1} | {type1} | {stats1} | {action1} |
| {sensor2} | {type2} | {stats2} | {action2} |

## 5. 실행 조치 (액션 플랜) ※체크리스트, 각 3항목 이내
### 5.1 즉시 (24시간)
- [ ] {immediate_1}. 근거: {imm_evidence1}. 기대효과: {imm_effect1}.
- [ ] {immediate_2}. 필요 자원: {imm_resource}.
- [ ] 가스·레이저·리코터 현장 점검. 로그 대조 필수.

### 5.2 단기 (1~2주)
- [ ] {short_1}. 검증: 시험 쿠폰/NDE.
- [ ] {short_2}. 지표: 불량률/다운타임 감소.

### 5.3 중장기 (1~3개월)
- [ ] {long_1}. ROI: {roi_note}.
- [ ] {long_2}. 단계별 적용 및 리스크 관리.

## 6. 결론
- 핵심 발견 1문장. (KPI 표 참조)
- 예상 영향 1문장. 생산/품질 관점.
- 우선 조치 1문장. 일정·책임 명시.
- 모니터링 계획 1문장. 핵심 지표·주기.

---
## 부록 A. 그래프 요약 (각 2문장)
### 그래프 1~10
각 그래프별: 목적/유형 + 핵심 증거 + 본문 연계

</REPORT_STRUCTURE>

<OUTPUT_FORMAT>
- 마크다운 형식 사용
- 표는 반드시 파이프(|) 형식으로 정렬
- 체크리스트는 - [ ] 형식
- 신호등 이모지: 🔴 (HIGH_RISK), 🟡 (MODERATE_RISK), 🟢 (HEALTHY)
- 섹션 구분 명확히 (##, ###)
- 그래프 인용 시 "(그래프 N 참조)" 형식만 사용
</OUTPUT_FORMAT>

<QUALITY_GUARDS>
- 원인 진단은 가설. 대안 가설 1개 병기.
- Self-consistency 불일치 시 MISMATCH 표기.
- 그래프 상세 묘사 금지(부록 외).
- 과도한 통계 설명 금지. AM 현상으로 번역.
- 수치 반올림. 단위 표기 필수. 재인용 금지.
</QUALITY_GUARDS>"""

# 간결형 보고서 프롬프트 (그래프 없음)
AM_BRIEF_EXPERT_PROMPT = """<ROLE>
역할: L-PBF/EBM/DED 등 적층제조(AM) 공정 품질진단 전문가
대상: 통계 비전문가인 현장 엔지니어
핵심 원칙: **통계 데이터를 장인의 암묵지처럼 해석**

문체: 개조식 단문, 명사형 종결, AM 용어 우선
</ROLE>

<AM_TERMINOLOGY>
- 리코터, 해치, 콘투어, 가스 퍼지, O₂ ppm, 스패터, 키홀, LOF
- SVD Mode → 공정 지배 패턴
- ICA Component → 독립 신호 분리 결과
- Energy Concentration → 에너지 집중도
- CV → 변동계수 (센서 출렁임 지표)
</AM_TERMINOLOGY>

<PROCESS_HEALTH_INTERPRETATION>
PROCESS_HEALTH 섹션 기반 판단:
- overall_status: HEALTHY/MODERATE_RISK/HIGH_RISK
- health_score: 0~1 범위 (≥0.85 양호, 0.60~0.85 주의, <0.60 위험)
- critical_issues: 즉시 조치 필요 항목
- warnings: 모니터링 필요 항목
- recommendation: 시스템 권장 조치
</PROCESS_HEALTH_INTERPRETATION>

<BRIEF_REPORT_STRUCTURE>
## 1. 개요
- 공정/데이터 요약
- **공정 건강: {overall_status} ({health_score}/1.00)**

## 2. 핵심 KPI 요약 (표)
| 항목 | 값 | AM 해석 |
|---|---:|---|

## 3. 공정 해석 (AM 관점)
### 3.1 가스·분위기
### 3.2 레이저·스캔
### 3.3 열·스테이지

## 4. 위험도 및 문제 센서 (표 2개)

## 5. 실행 조치 (체크리스트)
### 5.1 즉시 (24h)
### 5.2 단기 (1~2주)
### 5.3 중장기 (1~3개월)
</BRIEF_REPORT_STRUCTURE>

<SAFETY_GUARDS>
- 그래프 언급 금지 (간결형)
- 원인 단정 금지, 대안 가설 병기
- 데이터 부족 시 '판단 보류'
</SAFETY_GUARDS>"""


def check_requirements():
    """필수 패키지 확인"""
    missing_packages = []
    if genai is None:
        missing_packages.append("google-generativeai")
    if Image is None:
        missing_packages.append("Pillow")
    return missing_packages


def initialize_session_state() -> None:
    """세션 상태 초기화"""
    defaults = {
        'api_key': os.environ.get("API_KEY", ""),
        'model_name': "보통 (Flash)",
        'report_generated': False,
        'report_content': "",
        'analysis_type': "full",
        'parsed_health': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_process_health(content: str) -> Dict[str, Any]:
    """LLM_Ready_Report.txt에서 PROCESS_HEALTH 섹션 파싱"""
    health_data = {
        'overall_status': 'UNKNOWN',
        'health_score': 0.0,
        'energy_concentration_status': 'UNKNOWN',
        'mode1_energy_pct': 0.0,
        'category_balance_status': 'UNKNOWN',
        'critical_issues': [],
        'warnings': [],
        'recommendation': ''
    }

    # PROCESS_HEALTH 섹션 찾기
    health_match = re.search(r'=== PROCESS_HEALTH ===\n(.*?)(?:\n===|$)', content, re.DOTALL)
    if not health_match:
        return health_data

    health_section = health_match.group(1)

    # 각 필드 파싱
    patterns = {
        'overall_status': r'overall_status=(\w+)',
        'health_score': r'health_score=([\d.]+)',
        'energy_concentration_status': r'energy_concentration_status=(\w+)',
        'mode1_energy_pct': r'mode1_energy_pct=([\d.]+)',
        'category_balance_status': r'category_balance_status=(\w+)',
        'recommendation': r'recommendation=(.+?)(?:\n|$)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, health_section)
        if match:
            value = match.group(1)
            if key in ['health_score', 'mode1_energy_pct']:
                health_data[key] = float(value)
            else:
                health_data[key] = value

    # critical_issues 파싱
    critical_match = re.search(r'critical_issues:\n((?:  - .+\n)*)', health_section)
    if critical_match:
        issues = re.findall(r'  - (.+)', critical_match.group(1))
        health_data['critical_issues'] = issues

    # warnings 파싱
    warnings_match = re.search(r'warnings:\n((?:  - .+\n)*)', health_section)
    if warnings_match:
        warnings = re.findall(r'  - (.+)', warnings_match.group(1))
        health_data['warnings'] = warnings

    return health_data


def parse_ica_info(content: str) -> Dict[str, Any]:
    """ICA 분석 정보 파싱"""
    ica_data = {
        'total_components': 0,
        'problematic_count': 0,
        'problematic_ratio': 0.0
    }

    total_match = re.search(r'total_components=(\d+)', content)
    prob_match = re.search(r'problematic_count=(\d+)', content)

    if total_match:
        ica_data['total_components'] = int(total_match.group(1))
    if prob_match:
        ica_data['problematic_count'] = int(prob_match.group(1))

    if ica_data['total_components'] > 0:
        ica_data['problematic_ratio'] = (ica_data['problematic_count'] /
                                          ica_data['total_components']) * 100

    return ica_data


def get_health_status_display(status: str, score: float) -> tuple:
    """건강 상태에 따른 표시 정보 반환"""
    if status == 'HEALTHY':
        return ('🟢', 'health-healthy', '정상', '#4CAF50')
    elif status == 'MODERATE_RISK':
        return ('🟡', 'health-moderate', '주의', '#FF9800')
    else:  # HIGH_RISK
        return ('🔴', 'health-high-risk', '위험', '#f44336')


def run_inference(
    api_key: str,
    model_name: str,
    stats_content: str,
    images: Optional[List[Any]],
    prompt: str
) -> str:
    """AI API를 사용한 추론 실행"""

    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "max_output_tokens": 86384,
    }

    actual_model_name = MODEL_MAPPING.get(model_name, "gemini-2.5-flash")

    model = genai.GenerativeModel(
        model_name=actual_model_name,
        generation_config=generation_config,
        safety_settings={
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }
    )

    model_input = [
        prompt,
        "\n\n--- 분석 데이터 시작 ---\n",
        stats_content,
        "\n--- 분석 데이터 끝 ---\n",
    ]

    if images:
        model_input.append("\n--- 첨부 그래프 (10개) ---\n")
        model_input.extend(images)

    response = model.generate_content(model_input)
    return response.text


def main():
    initialize_session_state()

    # 헤더
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center'>
                <h1>🏭 AM 공정 분석 도구</h1>
                <p style='color: #7f8c8d; font-size: 1.1em'>
                    통계 데이터 → 장인의 암묵지 v3.0
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 패키지 확인
    missing = check_requirements()
    if missing:
        st.error(f"필수 패키지 미설치: {', '.join(missing)}")
        st.code(f"pip install {' '.join(missing)}")
        st.stop()

    # 사이드바 - API 설정 및 건강 상태 표시
    with st.sidebar:
        st.markdown("## API 설정")

        api_key = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            help="Google Generative AI API 키"
        )

        if api_key:
            st.session_state.api_key = api_key
            st.success("API 키 설정됨")

        model_name = st.selectbox(
            "모델 선택",
            options=list(MODEL_MAPPING.keys()),
            index=1,
            help="분석 복잡도에 따라 모델 선택"
        )
        st.session_state.model_name = model_name

        st.markdown("---")

        # 건강 상태 표시 (파싱된 경우)
        if st.session_state.parsed_health:
            health = st.session_state.parsed_health
            emoji, css_class, label, color = get_health_status_display(
                health['overall_status'], health['health_score']
            )

            st.markdown("### 공정 건강 상태")
            st.markdown(f"""
                <div class='{css_class}'>
                    <h2>{emoji} {health['overall_status']}</h2>
                    <h3>점수: {health['health_score']:.2f}/1.00</h3>
                </div>
            """, unsafe_allow_html=True)

            if health['critical_issues']:
                st.error("**Critical Issues:**")
                for issue in health['critical_issues']:
                    st.markdown(f"- {issue}")

            if health['warnings']:
                st.warning("**Warnings:**")
                for warn in health['warnings']:
                    st.markdown(f"- {warn}")

        st.markdown("---")
        st.markdown("### 분석 통계")
        if st.session_state.report_generated:
            st.metric("보고서 생성", "완료")
            st.metric("보고서 길이", f"{len(st.session_state.report_content):,} 자")
        else:
            st.info("보고서 생성 대기 중...")

    # 메인 컨텐츠 - 3개 탭
    tab1, tab2, tab3 = st.tabs([
        "📁 파일 업로드",
        "🚀 분석 실행",
        "📄 보고서 결과"
    ])

    # 탭 1: 파일 업로드
    with tab1:
        st.markdown("### 입력 데이터 업로드")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 LLM_Ready_Report.txt")
            stats_file = st.file_uploader(
                "통계 분석 결과 파일 선택",
                type=['txt', 'md', 'json'],
                help="PBF_SVD_LLM_Ready.py로 생성된 LLM_Ready_Report.txt"
            )

            if stats_file:
                st.success(f"파일 업로드 완료: {stats_file.name}")
                content = stats_file.read().decode('utf-8')
                stats_file.seek(0)

                # PROCESS_HEALTH 파싱
                health_data = parse_process_health(content)
                st.session_state.parsed_health = health_data

                # ICA 정보 파싱
                ica_data = parse_ica_info(content)

                # 파싱 결과 표시
                with st.expander("📊 파싱된 건강 상태", expanded=True):
                    emoji, _, label, color = get_health_status_display(
                        health_data['overall_status'], health_data['health_score']
                    )

                    cols = st.columns(3)
                    cols[0].metric("상태", f"{emoji} {label}")
                    cols[1].metric("점수", f"{health_data['health_score']:.2f}")
                    cols[2].metric("Mode1 에너지", f"{health_data['mode1_energy_pct']:.1f}%")

                    cols2 = st.columns(2)
                    cols2[0].metric("ICA 문제 비율", f"{ica_data['problematic_ratio']:.0f}%")
                    cols2[1].metric("에너지 집중", health_data['energy_concentration_status'])

                with st.expander("원본 파일 미리보기"):
                    st.text(content[:2000] + ("..." if len(content) > 2000 else ""))

        with col2:
            st.markdown("#### 📈 그래프 이미지 (선택)")
            graph_files = st.file_uploader(
                "그래프 이미지 10개 선택",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Plot 1~10 PNG 파일"
            )

            if graph_files:
                if len(graph_files) == 10:
                    st.success("그래프 10개 업로드 완료")
                    with st.expander("그래프 미리보기"):
                        cols = st.columns(5)
                        for i, file in enumerate(graph_files[:5]):
                            with cols[i]:
                                img = Image.open(file)
                                st.image(img, caption=f"그래프 {i+1}", use_container_width=True)
                        cols = st.columns(5)
                        for i, file in enumerate(graph_files[5:10]):
                            with cols[i]:
                                img = Image.open(file)
                                st.image(img, caption=f"그래프 {i+6}", use_container_width=True)
                else:
                    st.warning(f"10개 필요 (현재 {len(graph_files)}개)")
            else:
                st.info("💡 그래프 없이 간략 보고서 생성 가능")

    # 탭 2: 분석 실행
    with tab2:
        st.markdown("### 분석 실행")

        # 분석 유형 선택
        st.markdown("#### 분석 유형")
        analysis_type = st.radio(
            "분석 방식 선택",
            options=["full", "brief"],
            format_func=lambda x: "📊 전체 분석 (그래프 포함)" if x == "full" else "📝 간략 분석 (텍스트만)",
            horizontal=True
        )
        st.session_state.analysis_type = analysis_type

        if analysis_type == "full":
            st.info("10개 그래프 + 통계 데이터로 상세 보고서 생성")
        else:
            st.info("텍스트 데이터만으로 핵심 내용 위주 보고서 생성")

        st.markdown("---")

        # 입력 확인
        col1, col2, col3 = st.columns(3)
        with col1:
            api_ready = bool(st.session_state.api_key)
            st.metric("API 키", "✅" if api_ready else "❌")
        with col2:
            stats_ready = stats_file is not None
            st.metric("통계 데이터", "✅" if stats_ready else "❌")
        with col3:
            if analysis_type == "full":
                graphs_ready = graph_files and len(graph_files) == 10
                st.metric("그래프", "✅ 10개" if graphs_ready else "❌")
            else:
                graphs_ready = True
                st.metric("모드", "📝 간략")

        # 건강 상태 미리보기
        if st.session_state.parsed_health:
            health = st.session_state.parsed_health
            emoji, css_class, label, _ = get_health_status_display(
                health['overall_status'], health['health_score']
            )
            st.markdown(f"""
                <div class='{css_class}' style='margin: 1rem 0;'>
                    {emoji} 현재 공정 상태: <b>{health['overall_status']}</b>
                    (점수: {health['health_score']:.2f})
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 실행 버튼
        if st.button("🚀 분석 보고서 생성",
                    disabled=not (api_ready and stats_ready and graphs_ready),
                    use_container_width=True):

            with st.spinner("분석 중... (최대 2-3분 소요)"):
                try:
                    stats_content = stats_file.read().decode('utf-8')
                    stats_file.seek(0)

                    images = None
                    if analysis_type == "full" and graph_files:
                        images = []
                        for file in graph_files:
                            images.append(Image.open(file))
                            file.seek(0)

                    # 적절한 프롬프트 선택
                    if analysis_type == "full" and images:
                        use_prompt = AM_EXPERT_PROMPT
                    else:
                        use_prompt = AM_BRIEF_EXPERT_PROMPT

                    progress_bar = st.progress(0)
                    progress_bar.progress(30, text="모델 초기화 중...")

                    result = run_inference(
                        api_key=st.session_state.api_key,
                        model_name=st.session_state.model_name,
                        stats_content=stats_content,
                        images=images,
                        prompt=use_prompt
                    )

                    progress_bar.progress(90, text="보고서 생성 중...")

                    st.session_state.report_content = result
                    st.session_state.report_generated = True

                    progress_bar.progress(100, text="완료!")
                    time.sleep(0.5)
                    progress_bar.empty()

                    st.success("보고서 생성 완료!")
                    st.balloons()
                    st.info("📄 '보고서 결과' 탭에서 확인하세요")

                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")
                    if "quota" in str(e).lower():
                        st.warning("API 할당량 초과. 잠시 후 재시도.")
                    elif "api_key" in str(e).lower():
                        st.warning("API 키를 확인하세요.")

    # 탭 3: 보고서 결과
    with tab3:
        st.markdown("### 📄 생성된 보고서")

        if st.session_state.report_generated:
            # 메타데이터
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("생성 시간", datetime.now().strftime("%Y-%m-%d %H:%M"))
            with col2:
                st.metric("사용 모델", st.session_state.model_name)
            with col3:
                if st.session_state.parsed_health:
                    health = st.session_state.parsed_health
                    emoji, _, _, _ = get_health_status_display(
                        health['overall_status'], health['health_score']
                    )
                    st.metric("공정 상태", f"{emoji} {health['overall_status']}")

            st.markdown("---")

            # 보고서 내용
            with st.container():
                st.markdown(st.session_state.report_content)

            # 다운로드 버튼
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            with col1:
                # Markdown 파일
                health_info = ""
                if st.session_state.parsed_health:
                    h = st.session_state.parsed_health
                    health_info = f"\n**공정 상태:** {h['overall_status']} (점수: {h['health_score']:.2f})"

                report_md = f"""# AM 공정 분석 보고서

**생성 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**사용 모델:** {st.session_state.model_name}
**분석 유형:** {'전체 분석' if st.session_state.analysis_type == 'full' else '간략 분석'}{health_info}

---

{st.session_state.report_content}"""

                st.download_button(
                    label="📥 Markdown 다운로드",
                    data=report_md,
                    file_name=f"AM_Report_{timestamp}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            with col2:
                # JSON 파일
                json_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model": st.session_state.model_name,
                    "analysis_type": st.session_state.analysis_type,
                    "process_health": st.session_state.parsed_health,
                    "report": st.session_state.report_content
                }

                st.download_button(
                    label="📥 JSON 다운로드",
                    data=json.dumps(json_data, ensure_ascii=False, indent=2),
                    file_name=f"AM_Report_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col3:
                if st.button("🔄 새 분석", use_container_width=True):
                    st.session_state.report_generated = False
                    st.session_state.report_content = ""
                    st.session_state.parsed_health = None
                    st.rerun()
        else:
            st.info("📊 분석을 실행하면 여기에 보고서가 표시됩니다")
            st.markdown("""
            #### 사용 방법:
            1. **파일 업로드** 탭: LLM_Ready_Report.txt + 그래프 업로드
            2. **분석 실행** 탭: 분석 유형 선택 후 보고서 생성
            3. 생성된 보고서가 이 탭에 표시됩니다

            #### 새로운 기능 (v3.0):
            - **PROCESS_HEALTH 자동 파싱**: 건강 점수, 상태, 경고 자동 표시
            - **AM 전문가 용어 변환**: 통계 → 장인의 암묵지
            - **신호등 시스템**: 🟢 HEALTHY / 🟡 MODERATE_RISK / 🔴 HIGH_RISK
            """)


if __name__ == "__main__":
    main()
