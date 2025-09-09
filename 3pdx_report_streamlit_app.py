#!/usr/bin/env python3
"""
적층제조 공정 분석 애플리케이션 (Streamlit 버전)
통계 분석 결과와 그래프 이미지를 입력받아 전문 보고서를 생성합니다.
"""

import os
import streamlit as st
from datetime import datetime
from pathlib import Path
import time
from typing import List, Optional, Any, TYPE_CHECKING
import json

# 타입 체킹을 위한 조건부 임포트
if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType

# Pillow 라이브러리 임포트 시도
try:
    from PIL import Image
    from PIL.Image import Image as PILImage  # 타입 힌트용 별칭
except ImportError:
    Image = None
    PILImage = Any  # type: ignore

# google-generativeai 라이브러리 임포트 시도
try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore

# Streamlit 세션 스테이트 타입 정의
from typing import Protocol

class SessionState(Protocol):
    """Streamlit session state 타입 프로토콜"""
    api_key: str
    model_name: str
    report_generated: bool
    report_content: str
    analysis_type: str

# 페이지 설정
st.set_page_config(
    page_title="적층제조 공정 분석 도구",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS - 흰색 테마의 모던한 UI
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
    .uploadedFile {
        border: 2px dashed #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        background-color: #fafafa;
    }
    .report-container {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    h2 {
        color: #34495e;
        font-weight: 600;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #7f8c8d;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# 모델 매핑 딕셔너리
MODEL_MAPPING = {
    "간단": "gemini-2.5-flash-lite",
    "보통": "gemini-2.5-flash",
    "고급": "gemini-2.5-pro"
}



# 적층제조 분석 전문 프롬프트 (현장 간결판)
AM_ANALYSIS_PROMPT = """<ROLE>
- 역할: L-PBF 등 AM 공정 품질진단 전문가.
- 대상: 통계 비전문가인 현장 엔지니어.
- 문체: 공무원 개조식 단문. 명사형 종결. 불필요 수식어 금지.
- 용어: 통계용어 최소화. AM 터미놀러지 우선(리코터, 해치, 콘투어, 가스 퍼지, O₂ ppm, 스패터, 키홀, LOF, 에너지 밀도, 열 누적).
- 수치: 소수점 1~2자리. 단위 필수(분, %, 1/s, Hz, ppm).
- 편향 방지: 원인 단정 금지. 대안 가설 1개 병기. Confidence 표기.
</ROLE>

<SAFETY_GUARDS>
- 가설 언어 규칙: "의심", "가능성", "징후" 사용. "원인 확정" 금지.
- 데이터 부족·불일치 시: '판단 보류' 또는 '추가 확인 필요' 명시.
- 그래프는 번호(그래프 N)로만 인용. 본문 상세 묘사 금지(부록만 허용).
- KPI 수치 재인용 금지. 이후 'KPI 표 참조'로만 언급.
- 현장 안전 우선: 액션은 가역적·저비용·위험저감 순으로 제시.
</SAFETY_GUARDS>

<CONSISTENCY_RULES>
- Self-consistency(텍스트 vs 플롯) 필수. 각 항목을 MATCH/MISMATCH로 표기.
- MISMATCH 발생 시: 결론 강도 1단계 하향(예: '높음'→'중간'). 원인 1문장 기재.
- Confidence 산정(High/Med/Low):
  * High: (증거 ≥2종) AND (Self-consistency 대부분 MATCH) AND (동일 방향 지표 일치).
  * Med: 증거 ≥1종 AND 주요 지표 일부 불확실.
  * Low: 증거 부족 또는 MISMATCH 존재 또는 지표 상충.
</CONSISTENCY_RULES>

<DECISION_RULES>
- 위험 신호등(🔴/🟡/🟢) 기준(예시 가이드, 데이터에 맞춰 적용):
  * 🔴: 품질 점수 < 60 또는 DMD 불안정 모드 > 0 또는 다운타임 > 30분/빌드.
  * 🟡: 품질 점수 60~80 또는 SVD 이상률 3~10% 또는 사이클 CV 0.2~0.5.
  * 🟢: 품질 점수 ≥ 80 AND SVD 이상률 < 3% AND 사이클 CV < 0.2.
- 결론 강도 억제: 단일 지표로 중대한 결론 금지. 서로 다른 출처 2개 이상 합의 필요.
- 데이터 결손 처리: 값 미제공 시 '-' 출력. 결손 값으로 추론 금지.
</DECISION_RULES>

<RULES>
- 그래프는 본문에서 '증거 인용'만 수행. 상세 설명은 **부록 A**에 2문장 요약.
- 동일 수치 반복 금지. 최초 표만 제시. 이후 'KPI 표 참조'로 재인용.
- 모든 섹션은 불릿 위주. 각 절 최대 5불릿. 각 불릿 1문장.
- 6개 장 구조 고정. 제목·표 형식 고정. 변경 금지.
- Self-consistency(텍스트 vs 플롯) 필수 보고: MATCH/MISMATCH.
</RULES>

<REPORT_STRUCTURE>  # === 6개 장 고정 ===

## 1. 서론
- 공정: {process_type}. 장비/소재: {machine}/{material} (알려진 경우만).
- 목적: 빌드 안정성 점검 및 다운타임 원인 가설 도출.
- 데이터: 원본 {data_shape_original}, 처리 {data_shape_processed}, 해상도 {dt_seconds}s.
- 범위: 통계 신호 기반. 장비 이벤트 로그/현장 점검 미포함.
- Self-consistency: Stops {stops_text_vs_plot}, Downtime {downtime_text_vs_plot}, DMD {dmd_text_vs_plot}, SVD {svd_text_vs_plot}.

## 2. 핵심 지표(KPI) 요약  ※표 형식 고정
| 항목 | 값 | 단위 | 해석(AM 용어) |
|---|---:|:---:|---|
| 종합 품질 점수 | {quality_score:.1f} | /100 | 신호등 {risk_level} |
| SVD 유효 모드 | {significant_modes} | 개 | 공정 복잡도 판단 |
| 90% 에너지 컴포넌트 | {energy_90_components} | 개 | 지배 패턴 집중도 |
| DMD 불안정 모드 | {total_unstable_modes} | 개 | 성장 신호 존재 여부 |
| DMD 최대 성장률 | {max_growth_rate:.6f} | 1/s | 열 누적/진동 추정 |
| 사이클 CV | {coefficient_of_variation:.3f} | - | 빌드 리듬 변동도 |
| 정지 횟수 | {num_stops} | 회 | 계획/비계획 구분 필요 |
| 총 다운타임 | {total_stop_time_minutes:.1f} | 분 | 생산성 영향 |
| 이상률(SVD) | {svd_anomaly_rate:.2%} | - | 선형 이상 비율 |
| 이상치(IForest) | {iforest_anomalies} | 개 | 비선형 이상 지점 |
- 요약 판단: 공정 상태 총평 1문장.
- 주요 원인 가설 1문장 + 대안 가설 1문장.
- 즉시 조치 방향 1문장.

## 3. 공정 상태 해석(AM 관점)  ※그래프는 번호만 인용
### 3.1 가스·분위기(퍼지, O₂ ppm, 필터 ΔP)
- 증거: {gas_key_evidence}. (그래프 {gas_graph_ids} 참조)
- 해석: 보호가스 유지/스패터 제거 적정성 판단.
- 영향: 산화/결함(LOF·기공) 위험도 기술. Confidence={gas_conf}.

### 3.2 레이저·스캔(파워, 해치, 콘투어, LOT)
- 증거: {laser_key_evidence}. (그래프 {laser_graph_ids} 참조)
- 해석: 에너지 밀도/키홀·스패터 위험도 판단.
- 영향: 용융풀 안정/표면 조도/내부결함 영향. Confidence={laser_conf}.

### 3.3 열·스테이지(열 누적, 플랫폼, 리코터 인터랙션)
- 증거: {thermal_key_evidence}. (그래프 {thermal_graph_ids} 참조)
- 해석: 저주파 성장→열 누적 또는 리코터 간섭 추정.
- 영향: 변형/워핑/리코터 충돌 리스크. Confidence={thermal_conf}.

## 4. 위험도 평가(신호등)  ※표 형식 고정, 설명 최소화
| 순위 | 위험 요인(가설) | 영향도 | 근거 수치 | 관련 그래프 | 조치 우선 | Confidence |
|---:|---|---|---|---|---|---|
| 1 | {risk1} | 🔴/🟡/🟢 | {risk1_metrics} | {risk1_graphs} | 즉시 | {risk1_conf} |
| 2 | {risk2} | 🔴/🟡/🟢 | {risk2_metrics} | {risk2_graphs} | 1-2주 | {risk2_conf} |
| 3 | {risk3} | 🔴/🟡/🟢 | {risk3_metrics} | {risk3_graphs} | 1-2주 | {risk3_conf} |
| 4 | {risk4} | 🔴/🟡/🟢 | {risk4_metrics} | {risk4_graphs} | 정기 | {risk4_conf} |

### 4.2 문제 센서(상위)  ※표 형식 고정
| 센서 | 이상 유형 | 정량 근거 | 그래프 | 권장 조치 |
|---|---|---|---|---|
| {sensor1} | 변동성/드리프트/급변/상관 | {sensor1_stats} | {sensor1_graph} | 캘리브/점검/교체 |
| {sensor2} | ... | ... | ... | ... |
- 높은 상관 쌍 1줄 요약: [{s_pair1}] r={s_pair1_r}. 물리적 타당성 간단 기술.

## 5. 실행 조치(액션 플랜)  ※체크리스트, 각 3항목 이내
### 5.1 즉시(24시간)
- [ ] {immediate_action_1}. 근거: {imm_evidence}. 기대효과: {imm_effect}. Confidence={imm_conf}.
- [ ] {immediate_action_2}. 근거: {imm_evidence2}. 필요 자원: {imm_res}.
- [ ] 가스·레이저·리코터 현장 점검. 로그 대조 필수.

### 5.2 단기(1~2주)
- [ ] {short_term_1}. 검증: 시험 쿠폰/NDE.
- [ ] {short_term_2}. 지표: 불량률/다운타임 감소.

### 5.3 중장기(1~3개월)
- [ ] {long_term_1}. ROI: {roi_note}.
- [ ] {long_term_2}. 단계별 적용 및 리스크 관리.

## 6. 결론
- 핵심 발견 1문장. (KPI 표 참조)
- 예상 영향 1문장. 생산/품질 관점.
- 우선 조치 1문장. 일정·책임 명시.
- 모니터링 계획 1문장. 핵심 지표·주기.

---  # === 부록: 그래프는 여기서만 간단 설명 ===
## 부록 A. 그래프 1~10 요약(각 2문장 고정)
### 그래프 1
- 목적/유형: {g1_type_purpose}.
- 핵심 증거: {g1_key_msg}. 본문 연계: {g1_link}.
### 그래프 2
- 목적/유형: {g2_type_purpose}.
- 핵심 증거: {g2_key_msg}. 본문 연계: {g2_link}.
### 그래프 3
- 목적/유형: {g3_type_purpose}.
- 핵심 증거: {g3_key_msg}. 본문 연계: {g3_link}.
### 그래프 4
- 목적/유형: {g4_type_purpose}.
- 핵심 증거: {g4_key_msg}. 본문 연계: {g4_link}.
### 그래프 5
- 목적/유형: {g5_type_purpose}.
- 핵심 증거: {g5_key_msg}. 본문 연계: {g5_link}.
### 그래프 6
- 목적/유형: {g6_type_purpose}.
- 핵심 증거: {g6_key_msg}. 본문 연계: {g6_link}.
### 그래프 7
- 목적/유형: {g7_type_purpose}.
- 핵심 증거: {g7_key_msg}. 본문 연계: {g7_link}.
### 그래프 8
- 목적/유형: {g8_type_purpose}.
- 핵심 증거: {g8_key_msg}. 본문 연계: {g8_link}.
### 그래프 9
- 목적/유형: {g9_type_purpose}.
- 핵심 증거: {g9_key_msg}. 본문 연계: {g9_link}.
### 그래프 10
- 목적/유형: {g10_type_purpose}.
- 핵심 증거: {g10_key_msg}. 본문 연계: {g10_link}.

</REPORT_STRUCTURE>

<QUALITY_GUARDS>
- 원인 진단은 가설. 대안 가설 1개 병기. Confidence=High/Med/Low.
- Self-consistency 불일치 시 MISMATCH 표기. 원인 1문장 기술.
- 그래프 상세 묘사 금지(부록 외). 본문은 번호 인용만.
- 과도한 통계 설명 금지. AM 현상으로 번역하여 보고.
- 수치는 반올림. 단위 표기 누락 금지. 재인용 금지.
</QUALITY_GUARDS>

<INPUT_BINDINGS>  # 스크립트가 제공해야 할 필드. 미제공 시 '-' 출력.
required: [
  process_type, machine, material, data_shape_original, data_shape_processed, dt_seconds,
  stops_text_vs_plot, downtime_text_vs_plot, dmd_text_vs_plot, svd_text_vs_plot,
  quality_score, risk_level, significant_modes, energy_90_components, total_unstable_modes,
  max_growth_rate, coefficient_of_variation, num_stops, total_stop_time_minutes,
  svd_anomaly_rate, iforest_anomalies,
  gas_key_evidence, gas_graph_ids, gas_conf,
  laser_key_evidence, laser_graph_ids, laser_conf,
  thermal_key_evidence, thermal_graph_ids, thermal_conf,
  risk1, risk1_metrics, risk1_graphs, risk1_conf,
  risk2, risk2_metrics, risk2_graphs, risk2_conf,
  risk3, risk3_metrics, risk3_graphs, risk3_conf,
  risk4, risk4_metrics, risk4_graphs, risk4_conf,
  sensor1, sensor1_stats, sensor1_graph, s_pair1, s_pair1_r,
  immediate_action_1, imm_evidence, imm_effect, imm_conf, immediate_action_2, imm_evidence2, imm_res,
  short_term_1, short_term_2, long_term_1, long_term_2, roi_note,
  g1_type_purpose, g1_key_msg, g1_link, g2_type_purpose, g2_key_msg, g2_link,
  g3_type_purpose, g3_key_msg, g3_link, g4_type_purpose, g4_key_msg, g4_link,
  g5_type_purpose, g5_key_msg, g5_link, g6_type_purpose, g6_key_msg, g6_link,
  g7_type_purpose, g7_key_msg, g7_link, g8_type_purpose, g8_key_msg, g8_link,
  g9_type_purpose, g9_key_msg, g9_link, g10_type_purpose, g10_key_msg, g10_link,
  timestamp, input_file
]
fallback_rule: "값이 없으면 '-' 또는 '판단 보류' 출력. 추정 금지."
</INPUT_BINDINGS>

<OUTPUT_HEADER>  # 자동 머리말(고정)
분석 완료: {timestamp}
입력 파일: {input_file}
그래프: 10개(그래프 1~10)
</OUTPUT_HEADER>"""

# 적층제조 간결형 보고서 프롬프트 (그래프 제외 · 형식 고정 · 현장 친화)
AM_BRIEF_PROMPT = """<ROLE>
- 역할: L-PBF 등 AM 공정 품질진단 전문가.
- 대상: 통계 비전문가인 현장 엔지니어.
- 문체: 공무원 개조식 단문. 명사형 종결. 불필요 수식어 금지.
- 용어: 통계용어 최소화. AM 터미놀러지 우선(리코터, 해치, 콘투어, 가스 퍼지, O₂ ppm, 스패터, 키홀, LOF, 에너지 밀도, 열 누적).
- 수치: 소수점 1~2자리. 단위 필수(분, %, 1/s, Hz, ppm).
- 편향 방지: 원인 단정 금지. 대안 가설 1개 병기. Confidence 표기(High/Med/Low).
</ROLE>

<SAFETY_GUARDS>
- 그래프·그림·캡처 금지. 표와 텍스트만 사용.
- 단일 지표로 중대한 결론 금지. 최소 2개 출처 합의 필요(SVD/DMD/IForest/사이클).
- Self-consistency 필수: 지표 간 상충 시 MISMATCH 표기 및 결론 강도 1단계 하향.
- 데이터 결손 시 추정 금지. '-' 또는 '판단 보류' 출력.
- 현장 안전 우선: 액션은 가역적·저비용·위험저감 순으로 제시.
</SAFETY_GUARDS>

<DECISION_RULES>
- 위험 신호등(🔴/🟡/🟢) 기준(가이드):
  * 🔴: 품질 점수 < 60 또는 DMD 불안정 모드 > 0 또는 총 다운타임 > 30분/빌드.
  * 🟡: 품질 점수 60~80 또는 SVD 이상률 3~10% 또는 사이클 CV 0.2~0.5.
  * 🟢: 품질 점수 ≥ 80 AND SVD 이상률 < 3% AND 사이클 CV < 0.2.
- Confidence 산정:
  * High: 증거 ≥2종 일치 AND Self-consistency=대부분 MATCH.
  * Med: 증거 ≥1종 일치 또는 일부 상충.
  * Low: 증거 부족 또는 MISMATCH 존재.
</DECISION_RULES>

<RULES>
- 5개 장 구조 고정. 제목·표 형식 고정. 변경 금지.
- 각 절 최대 5불릿. 각 불릿 1문장.
- 동일 수치 반복 금지. 최초 표만 제시. 이후 'KPI 표 참조'로 재인용.
- 단위 누락 금지. 반올림(소수 1~2자리).
</RULES>

<REPORT_STRUCTURE>  # === 5개 장 고정 ===

<OUTPUT_HEADER>  # 자동 머리말(고정)
분석 완료: {timestamp}
입력 파일: {input_file}
그래프: 포함 안함(간결형)
</OUTPUT_HEADER>

## 1. 개요
- 공정: {process_type}. 장비/소재: {machine}/{material} (알려진 경우만).
- 목적: 빌드 안정성 점검 및 다운타임 원인 가설 도출.
- 데이터: 원본 {data_shape_original}, 처리 {data_shape_processed}, 해상도 {dt_seconds}s, 윈도우 {window_size}s.
- 범위: 통계 신호 기반. 장비 이벤트 로그/현장 점검 미포함.
- Self-consistency 요약: Stops {stops_consistency}, DMD {dmd_consistency}, SVD {svd_consistency}. 상충 시 결론 강도 하향.

## 2. 핵심 지표(KPI) 요약  ※표 형식 고정
| 항목 | 값 | 단위 | 해석(AM 용어) |
|---|---:|:---:|---|
| 종합 품질 점수 | {quality_score:.1f} | /100 | 신호등 {risk_level} |
| SVD 유효 모드 | {significant_modes} | 개 | 공정 복잡도 판단 |
| 90% 에너지 컴포넌트 | {energy_90_components} | 개 | 지배 패턴 집중도 |
| DMD 불안정 모드 | {total_unstable_modes} | 개 | 성장 신호 존재 여부 |
| DMD 최대 성장률 | {max_growth_rate:.6f} | 1/s | 열 누적/진동 추정 |
| 사이클 CV | {coefficient_of_variation:.3f} | - | 빌드 리듬 변동도 |
| 정지 횟수 | {num_stops} | 회 | 계획/비계획 구분 필요 |
| 총 다운타임 | {total_stop_time_minutes:.1f} | 분 | 생산성 영향 |
| 이상률(SVD) | {svd_anomaly_rate:.2%} | - | 선형 이상 비율 |
| 이상치(IForest) | {iforest_anomalies} | 개 | 비선형 이상 지점 |
- 요약 판단: 공정 상태 총평 1문장.
- 주요 원인 가설 1문장 + 대안 가설 1문장. Confidence={root_cause_conf}.
- 즉시 조치 방향 1문장.

## 3. 공정 상태 해석(AM 관점)  ※텍스트만, 표 인용 중심
### 3.1 가스·분위기(퍼지, O₂ ppm, 필터 ΔP)
- 증거: {gas_key_evidence_txt}. (KPI 표 참조)
- 해석: 보호가스 유지/스패터 제거 적정성 판단.
- 영향: 산화/결함(LOF·기공) 위험도 기술. Confidence={gas_conf}.

### 3.2 레이저·스캔(파워, 해치, 콘투어, LOT)
- 증거: {laser_key_evidence_txt}. (KPI 표 참조)
- 해석: 에너지 밀도/키홀·스패터 위험도 판단.
- 영향: 용융풀 안정/표면 조도/내부결함 영향. Confidence={laser_conf}.

### 3.3 열·스테이지(열 누적, 플랫폼, 리코터 인터랙션)
- 증거: {thermal_key_evidence_txt}. (KPI 표 참조)
- 해석: 저주파 성장→열 누적 또는 리코터 간섭 추정.
- 영향: 변형/워핑/리코터 충돌 리스크. Confidence={thermal_conf}.

## 4. 위험도 요약 및 문제 센서  ※표 2개 고정
### 4.1 위험도 요약(신호등)
| 순위 | 위험 요인(가설) | 영향도 | 근거 수치 | 조치 우선 | Confidence |
|---:|---|---|---|---|---|
| 1 | {risk1} | 🔴/🟡/🟢 | {risk1_metrics} | 즉시 | {risk1_conf} |
| 2 | {risk2} | 🔴/🟡/🟢 | {risk2_metrics} | 1-2주 | {risk2_conf} |
| 3 | {risk3} | 🔴/🟡/🟢 | {risk3_metrics} | 1-2주 | {risk3_conf} |

### 4.2 문제 센서(상위)
| 센서 | 이상 유형 | 정량 근거 | 권장 조치 |
|---|---|---|---|
| {sensor1} | 변동성/드리프트/급변/상관 | {sensor1_stats} | 캘리브/점검/교체 |
| {sensor2} | ... | ... | ... |
- 높은 상관 쌍 1줄 요약: [{s_pair1}] r={s_pair1_r}. 물리적 타당성 간단 기술.

## 5. 실행 조치(액션 플랜)  ※체크리스트, 각 3항목 이내
### 5.1 즉시(24시간)
- [ ] {immediate_action_1}. 근거: {imm_evidence}. 기대효과: {imm_effect}. Confidence={imm_conf}.
- [ ] {immediate_action_2}. 근거: {imm_evidence2}. 필요 자원: {imm_res}.
- [ ] 가스·레이저·리코터 현장 점검. 로그 대조 필수.

### 5.2 단기(1~2주)
- [ ] {short_term_1}. 검증: 시험 쿠폰/NDE.
- [ ] {short_term_2}. 지표: 불량률/다운타임 감소.

### 5.3 중장기(1~3개월)
- [ ] {long_term_1}. ROI: {roi_note}.
- [ ] {long_term_2}. 단계별 적용 및 리스크 관리.

</REPORT_STRUCTURE>

<QUALITY_GUARDS>
- 원인 진단은 가설. 대안 가설 1개 병기. Confidence=High/Med/Low.
- 그래프·이미지 언급 금지. 'KPI 표 참조'로만 재인용.
- Self-consistency 불일치 시 MISMATCH 표기. 원인 1문장 기술.
- 과도한 통계 설명 금지. AM 현상으로 번역하여 보고.
- 수치 재사용 금지. 표 외 반복 시 'KPI 표 참조'로 대체.
</QUALITY_GUARDS>

<INPUT_BINDINGS>  # 스크립트가 제공해야 할 필드. 미제공 시 '-' 출력.
required: [
  timestamp, input_file,
  process_type, machine, material, data_shape_original, data_shape_processed, dt_seconds, window_size,
  quality_score, risk_level, significant_modes, energy_90_components, total_unstable_modes,
  max_growth_rate, coefficient_of_variation, num_stops, total_stop_time_minutes,
  svd_anomaly_rate, iforest_anomalies,
  stops_consistency, dmd_consistency, svd_consistency,
  gas_key_evidence_txt, gas_conf,
  laser_key_evidence_txt, laser_conf,
  thermal_key_evidence_txt, thermal_conf,
  risk1, risk1_metrics, risk1_conf,
  risk2, risk2_metrics, risk2_conf,
  risk3, risk3_metrics, risk3_conf,
  sensor1, sensor1_stats, s_pair1, s_pair1_r,
  immediate_action_1, imm_evidence, imm_effect, imm_conf,
  immediate_action_2, imm_evidence2, imm_res,
  short_term_1, short_term_2, long_term_1, long_term_2, roi_note
]
fallback_rule: "값이 없으면 '-' 또는 '판단 보류' 출력. 추정 금지."
</INPUT_BINDINGS>"""


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
    # getattr를 사용한 안전한 접근
    if not hasattr(st.session_state, 'api_key'):
        st.session_state.api_key = os.environ.get("API_KEY", "")
    if not hasattr(st.session_state, 'model_name'):
        st.session_state.model_name = "보통"
    if not hasattr(st.session_state, 'report_generated'):
        st.session_state.report_generated = False
    if not hasattr(st.session_state, 'report_content'):
        st.session_state.report_content = ""
    if not hasattr(st.session_state, 'analysis_type'):
        st.session_state.analysis_type = "full"

def run_inference(
    api_key: str, 
    model_name: str, 
    stats_content: str, 
    images: Optional[List[Any]],  # 또는 Optional[List['PILImage']]
    prompt: str
) -> str:
    """AI API를 사용한 추론 실행"""
    
    # API 설정
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 20,
        "max_output_tokens": 86384,
    }
    
    # 실제 모델명으로 변환
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
    
    # 모델 입력 구성
    model_input = [
        prompt,
        "\n\n--- 분석 데이터 시작 ---\n",
        stats_content,
        "\n--- 분석 데이터 끝 ---\n",
    ]
    
    # 이미지가 있으면 추가
    if images:
        model_input.append("\n--- 첨부 그래프 (10개) ---\n")
        model_input.extend(images)
    
    # API 호출
    response = model.generate_content(model_input)
    return response.text

def main():
    # 세션 상태 초기화
    initialize_session_state()
    
    # 헤더
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center'>
                <h1>🏭 적층제조 공정 분석 도구</h1>
                <p style='color: #7f8c8d; font-size: 1.1em'>AI 기반 보고서 생성 시스템 v2.0</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 패키지 확인
    missing = check_requirements()
    if missing:
        st.error(f"⚠️ 필수 패키지가 설치되지 않았습니다: {', '.join(missing)}")
        st.code(f"pip install {' '.join(missing)}")
        st.stop()
    
    # 사이드바 - API 설정
    with st.sidebar:
        st.markdown("## ⚙️ API 설정")
        
        api_key = st.text_input(
            "API Key 입력",
            value=st.session_state.api_key,
            type="password",
            help="AI 서비스 API 키를 입력하세요"
        )
        
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API 키 설정됨")
        
        model_name = st.selectbox(
            "모델 선택",
            options=["간단", "보통", "고급"],
            index=1,
            help="분석 복잡도에 따라 모델을 선택하세요"
        )
        st.session_state.model_name = model_name
        
        st.markdown("---")
        st.markdown("### 📊 분석 통계")
        if st.session_state.report_generated:
            st.metric("보고서 생성", "완료 ✅")
            st.metric("보고서 길이", f"{len(st.session_state.report_content)} 자")
        else:
            st.info("보고서 생성 대기 중...")
    
    # 메인 컨텐츠
    tab1, tab2, tab3 = st.tabs(["📁 파일 업로드", "🚀 분석 실행", "📄 보고서 결과"])
    
    with tab1:
        st.markdown("### 입력 데이터 업로드")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 통계 데이터 파일")
            stats_file = st.file_uploader(
                "통계 분석 결과 파일을 선택하세요",
                type=['txt', 'md', 'json'],
                help="통계 분석 결과가 담긴 텍스트 파일을 업로드하세요"
            )
            
            if stats_file:
                st.success(f"✅ 파일 업로드 완료: {stats_file.name}")
                with st.expander("파일 내용 미리보기"):
                    content = stats_file.read().decode('utf-8')
                    st.text(content[:1000] + ("..." if len(content) > 1000 else ""))
                    stats_file.seek(0)  # 파일 포인터 리셋
        
        with col2:
            st.markdown("#### 📈 그래프 이미지 파일 (선택사항)")
            graph_files = st.file_uploader(
                "그래프 이미지 10개를 선택하세요",
                type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
                accept_multiple_files=True,
                help="정확히 10개의 그래프 파일을 선택해야 합니다 (선택사항)"
            )
            
            if graph_files:
                if len(graph_files) == 10:
                    st.success(f"✅ 그래프 10개 업로드 완료")
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
                    st.warning(f"⚠️ 10개의 그래프가 필요합니다 (현재 {len(graph_files)}개)")
            else:
                st.info("💡 그래프 없이 텍스트만으로 간략 보고서 생성이 가능합니다")
    
    with tab2:
        st.markdown("### 분석 실행")
        
        # 분석 유형 선택
        st.markdown("#### 분석 유형 선택")
        analysis_type = st.radio(
            "분석 방식을 선택하세요",
            options=["full", "brief"],
            format_func=lambda x: "📊 전체 분석 (그래프 포함)" if x == "full" else "📝 간략 분석 (텍스트만)",
            horizontal=True,
            help="그래프가 없는 경우 자동으로 간략 분석이 선택됩니다"
        )
        st.session_state.analysis_type = analysis_type
        
        if analysis_type == "full":
            st.info("📊 10개의 그래프와 통계 데이터를 함께 분석하여 상세한 보고서를 생성합니다")
        else:
            st.info("📝 텍스트 데이터만으로 핵심 내용 위주의 간략한 보고서를 생성합니다")
        
        st.markdown("---")
        
        # 입력 확인
        col1, col2, col3 = st.columns(3)
        with col1:
            api_ready = bool(st.session_state.api_key)
            st.metric("API 키", "✅ 설정됨" if api_ready else "❌ 미설정")
        with col2:
            stats_ready = stats_file is not None
            st.metric("통계 데이터", "✅ 준비됨" if stats_ready else "❌ 미업로드")
        with col3:
            if st.session_state.analysis_type == "full":
                graphs_ready = graph_files and len(graph_files) == 10
                st.metric("그래프 파일", "✅ 10개 준비" if graphs_ready else "❌ 미준비")
            else:
                graphs_ready = True  # 간략 분석은 그래프 불필요
                st.metric("분석 모드", "📝 간략 분석")
        
        # 실행 버튼
        st.markdown("---")
        
        if st.button("🚀 분석 보고서 생성", 
                    disabled=not (api_ready and stats_ready and graphs_ready),
                    use_container_width=True):
            
            with st.spinner("분석 중... (최대 2-3분 소요)"):
                try:
                    # 통계 데이터 읽기
                    stats_content = stats_file.read().decode('utf-8')
                    stats_file.seek(0)
                    
                    # 이미지 로드 (전체 분석인 경우)
                    images = None
                    if st.session_state.analysis_type == "full" and graph_files:
                        images = []
                        for file in graph_files:
                            images.append(Image.open(file))
                            file.seek(0)
                    
                    # 적절한 프롬프트 선택
                    if st.session_state.analysis_type == "full" and images:
                        use_prompt = AM_ANALYSIS_PROMPT
                    else:
                        use_prompt = AM_BRIEF_PROMPT
                    
                    # API 호출
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
                    
                    # 결과 저장
                    st.session_state.report_content = result
                    st.session_state.report_generated = True
                    
                    progress_bar.progress(100, text="완료!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    
                    st.success("✅ 보고서 생성 완료!")
                    st.balloons()
                    
                    # 자동으로 결과 탭으로 이동하는 효과
                    st.info("📄 '보고서 결과' 탭에서 생성된 보고서를 확인하세요")
                    
                except Exception as e:
                    st.error(f"❌ 오류 발생: {str(e)}")
                    if "quota" in str(e).lower():
                        st.warning("💡 API 할당량이 초과되었습니다. 잠시 후 다시 시도하세요.")
                    elif "api_key" in str(e).lower():
                        st.warning("💡 API 키를 확인하세요.")
    
    with tab3:
        st.markdown("### 📄 생성된 보고서")
        
        if st.session_state.report_generated:
            # 보고서 메타데이터
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("생성 시간", datetime.now().strftime("%Y-%m-%d %H:%M"))
            with col2:
                st.metric("사용 모델", st.session_state.model_name)
            with col3:
                st.metric("분석 유형", "전체 분석" if st.session_state.analysis_type == "full" else "간략 분석")
            
            st.markdown("---")
            
            # 보고서 내용 표시
            with st.container():
                st.markdown(st.session_state.report_content)
            
            # 다운로드 버튼
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                # Markdown 파일로 다운로드
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"AM_analysis_report_{timestamp}.md"
                
                report_with_header = f"""# 적층제조 공정 분석 보고서
                
**생성 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**분석 유형:** {'전체 분석 (그래프 포함)' if st.session_state.analysis_type == 'full' else '간략 분석 (텍스트만)'}

---

{st.session_state.report_content}"""
                
                st.download_button(
                    label="📥 Markdown 파일 다운로드",
                    data=report_with_header,
                    file_name=filename,
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                # JSON 형식으로 다운로드
                json_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model": st.session_state.model_name,
                    "analysis_type": st.session_state.analysis_type,
                    "report": st.session_state.report_content
                }
                
                st.download_button(
                    label="📥 JSON 파일 다운로드",
                    data=json.dumps(json_data, ensure_ascii=False, indent=2),
                    file_name=f"AM_analysis_report_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # 새 분석 버튼
            if st.button("🔄 새로운 분석 시작", use_container_width=True):
                st.session_state.report_generated = False
                st.session_state.report_content = ""
                st.rerun()
        else:
            st.info("📊 분석을 실행하면 여기에 보고서가 표시됩니다")
            st.markdown("""
            #### 사용 방법:
            1. **파일 업로드** 탭에서 통계 데이터와 그래프 파일 업로드
            2. **분석 실행** 탭에서 분석 유형 선택 후 보고서 생성 버튼 클릭
            3. 생성된 보고서가 이 탭에 표시됩니다
            """)

if __name__ == "__main__":
    main()
