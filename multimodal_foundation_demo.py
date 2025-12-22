#!/usr/bin/env python3
"""
멀티모달 파운데이션 모델 데모 애플리케이션
- Thingiverse 3D 데이터 기반 멀티모달 학습 시연
- 3D 모델 + 이미지 + 텍스트 통합 분석
- 대상: 엔지니어링 디자인과 사무관, 3D 피아 팀

개발: AM 센터
버전: 1.0
"""

import streamlit as st
import numpy as np
from datetime import datetime
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import io
import base64

# Pillow 라이브러리 임포트
try:
    from PIL import Image
except ImportError:
    Image = None

# Plotly 3D 시각화
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="멀티모달 파운데이션 모델 Demo",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }

    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }

    .modal-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }

    .embedding-viz {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }

    .similarity-high { background-color: #4CAF50; color: white; padding: 0.3rem 0.6rem; border-radius: 4px; }
    .similarity-med { background-color: #FF9800; color: white; padding: 0.3rem 0.6rem; border-radius: 4px; }
    .similarity-low { background-color: #9E9E9E; color: white; padding: 0.3rem 0.6rem; border-radius: 4px; }

    .feature-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
    }

    .info-card {
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 샘플 Thingiverse 스타일 데이터
# ============================================================

SAMPLE_DESIGNS = [
    {
        "id": "thing_001",
        "name": "Parametric Gear Set",
        "category": "Engineering",
        "tags": ["gear", "parametric", "mechanical", "assembly"],
        "description": "완전 파라메트릭 기어 세트. OpenSCAD로 설계. 모듈, 톱니 수, 압력각 조절 가능.",
        "author": "MechEngineer",
        "downloads": 15420,
        "likes": 892,
        "material": "PLA, PETG",
        "print_time": "2-4시간",
        "embedding": np.random.randn(128).tolist()
    },
    {
        "id": "thing_002",
        "name": "Turbine Blade Test Specimen",
        "category": "Aerospace",
        "tags": ["turbine", "aerospace", "test", "specimen"],
        "description": "항공엔진 터빈 블레이드 시험편. 크리프 테스트용 표준 형상. Ti-6Al-4V 최적화.",
        "author": "AeroResearch",
        "downloads": 8930,
        "likes": 567,
        "material": "Ti-6Al-4V, Inconel 718",
        "print_time": "6-8시간",
        "embedding": np.random.randn(128).tolist()
    },
    {
        "id": "thing_003",
        "name": "Lattice Structure Cube",
        "category": "Research",
        "tags": ["lattice", "lightweight", "topology", "optimization"],
        "description": "경량화 격자 구조 큐브. BCC/Gyroid/Octet 선택 가능. 압축 시험용.",
        "author": "StructureLab",
        "downloads": 12100,
        "likes": 743,
        "material": "AlSi10Mg, SS316L",
        "print_time": "4-6시간",
        "embedding": np.random.randn(128).tolist()
    },
    {
        "id": "thing_004",
        "name": "Heat Exchanger Channel",
        "category": "Thermal",
        "tags": ["heat exchanger", "channel", "thermal", "fluid"],
        "description": "적층제조 최적화 열교환기 채널. TPMS 기반 내부 구조. CFD 검증 완료.",
        "author": "ThermalDesign",
        "downloads": 6780,
        "likes": 421,
        "material": "CuCrZr, Inconel",
        "print_time": "8-12시간",
        "embedding": np.random.randn(128).tolist()
    },
    {
        "id": "thing_005",
        "name": "Medical Implant Scaffold",
        "category": "Medical",
        "tags": ["implant", "scaffold", "biomedical", "porous"],
        "description": "다공성 의료용 임플란트 스캐폴드. 골 유입 최적화 설계. ISO 13485 고려.",
        "author": "BioMedDev",
        "downloads": 4560,
        "likes": 389,
        "material": "Ti-6Al-4V ELI",
        "print_time": "3-5시간",
        "embedding": np.random.randn(128).tolist()
    }
]


# ============================================================
# 유틸리티 함수
# ============================================================

def generate_mock_embedding(text: str, dim: int = 128) -> np.ndarray:
    """텍스트 기반 의사 임베딩 생성 (해시 기반)"""
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()

    np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
    embedding = np.random.randn(dim)
    return embedding / np.linalg.norm(embedding)


def generate_image_embedding(img_array: np.ndarray, dim: int = 128) -> np.ndarray:
    """이미지 기반 의사 임베딩 생성"""
    # 이미지 특성에서 시드 생성
    seed = int(np.sum(img_array) % (2**31))
    np.random.seed(seed)
    embedding = np.random.randn(dim)
    return embedding / np.linalg.norm(embedding)


def generate_3d_embedding(vertices: np.ndarray, dim: int = 128) -> np.ndarray:
    """3D 형상 기반 의사 임베딩 생성"""
    # 3D 특성에서 시드 생성
    seed = int(np.sum(vertices) % (2**31))
    np.random.seed(seed)
    embedding = np.random.randn(dim)
    return embedding / np.linalg.norm(embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도 계산"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def parse_stl_ascii(content: str) -> Tuple[np.ndarray, np.ndarray]:
    """ASCII STL 파일 파싱 (간단 버전)"""
    vertices = []
    normals = []

    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('vertex'):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith('facet normal'):
            parts = line.split()
            if len(parts) >= 5:
                normals.append([float(parts[2]), float(parts[3]), float(parts[4])])

    return np.array(vertices), np.array(normals)


def create_sample_3d_model(model_type: str = "cube") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """샘플 3D 모델 생성"""
    if model_type == "gear":
        # 기어 형상 (간단화)
        n_teeth = 12
        r_outer = 1.0
        r_inner = 0.8

        theta = np.linspace(0, 2*np.pi, n_teeth*4, endpoint=False)
        r = np.where(np.arange(len(theta)) % 4 < 2, r_outer, r_inner)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z_bottom = np.zeros_like(x)
        z_top = np.ones_like(x) * 0.3

        vertices = np.vstack([
            np.column_stack([x, y, z_bottom]),
            np.column_stack([x, y, z_top])
        ])

        # 삼각형 인덱스 생성
        n = len(theta)
        faces = []
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([i, next_i, i + n])
            faces.append([next_i, next_i + n, i + n])

        return vertices, np.array(faces), np.zeros((len(faces), 3))

    elif model_type == "lattice":
        # 격자 구조
        size = 10
        points = []
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    if (x + y + z) % 2 == 0:
                        points.append([x*0.1, y*0.1, z*0.1])

        vertices = np.array(points)
        faces = np.array([[0, 1, 2]])  # 단순화
        return vertices, faces, np.zeros((1, 3))

    else:  # cube
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 5, 1], [0, 4, 5],  # front
            [2, 7, 3], [2, 6, 7],  # back
            [0, 7, 4], [0, 3, 7],  # left
            [1, 6, 2], [1, 5, 6]   # right
        ])
        return vertices, faces, np.zeros((len(faces), 3))


def visualize_3d_mesh(vertices: np.ndarray, faces: np.ndarray = None) -> go.Figure:
    """Plotly로 3D 메시 시각화"""
    if not PLOTLY_AVAILABLE:
        return None

    if faces is not None and len(faces) > 0:
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0] if len(faces.shape) > 1 else None,
                j=faces[:, 1] if len(faces.shape) > 1 else None,
                k=faces[:, 2] if len(faces.shape) > 1 else None,
                colorscale='Viridis',
                intensity=vertices[:, 2],
                opacity=0.8
            )
        ])
    else:
        # 점 구름으로 표시
        fig = go.Figure(data=[
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=vertices[:, 2],
                    colorscale='Viridis',
                    opacity=0.8
                )
            )
        ])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )

    return fig


def visualize_embedding(embeddings: Dict[str, np.ndarray]) -> go.Figure:
    """임베딩 시각화 (t-SNE 스타일 2D 투영 시뮬레이션)"""
    if not PLOTLY_AVAILABLE:
        return None

    colors = {'text': '#667eea', 'image': '#f5576c', '3d': '#11998e', 'fused': '#f093fb'}

    fig = go.Figure()

    for name, emb in embeddings.items():
        # 임베딩을 2D로 투영 (간단한 PCA 시뮬레이션)
        np.random.seed(hash(name) % (2**31))
        x = np.sum(emb[:64]) + np.random.randn() * 0.1
        y = np.sum(emb[64:]) + np.random.randn() * 0.1

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            name=name.upper(),
            text=[name.upper()],
            textposition='top center',
            marker=dict(
                size=30,
                color=colors.get(name, '#999'),
                symbol='circle'
            )
        ))

    fig.update_layout(
        title='멀티모달 임베딩 공간',
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        height=350,
        showlegend=True
    )

    return fig


def search_similar_designs(query_embedding: np.ndarray, designs: List[Dict], top_k: int = 3) -> List[Dict]:
    """유사 디자인 검색"""
    results = []

    for design in designs:
        design_emb = np.array(design['embedding'])
        similarity = cosine_similarity(query_embedding, design_emb)
        results.append({
            **design,
            'similarity': similarity
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]


# ============================================================
# 세션 상태 초기화
# ============================================================

def initialize_session_state():
    defaults = {
        'text_input': "",
        'image_uploaded': False,
        'stl_uploaded': False,
        'embeddings': {},
        'search_results': [],
        'demo_mode': 'explore'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# 메인 앱
# ============================================================

def main():
    initialize_session_state()

    # 헤더
    st.markdown("""
    <div class='hero-section'>
        <h1>🔮 멀티모달 파운데이션 모델</h1>
        <p style='font-size: 1.2em'>3D 모델 + 이미지 + 텍스트 = 통합 AI 이해</p>
        <p style='font-size: 0.9em; opacity: 0.8'>Thingiverse 데이터 기반 엔지니어링 디자인 AI</p>
    </div>
    """, unsafe_allow_html=True)

    # 사이드바 - 모드 선택 및 정보
    with st.sidebar:
        st.markdown("## 🎯 데모 모드")
        demo_mode = st.radio(
            "모드 선택",
            options=['explore', 'embed', 'search', 'learn'],
            format_func=lambda x: {
                'explore': '🗂️ 데이터 탐색',
                'embed': '🧬 임베딩 시연',
                'search': '🔍 유사 검색',
                'learn': '📚 학습 시뮬레이션'
            }[x],
            key='demo_mode'
        )

        st.markdown("---")

        st.markdown("### 📊 멀티모달 구성")
        st.markdown("""
        <div class='info-card'>
        <b>3가지 모달리티:</b><br>
        • 🎨 <b>3D 형상</b>: STL/OBJ 기하 정보<br>
        • 🖼️ <b>이미지</b>: 렌더링/실물 사진<br>
        • 📝 <b>텍스트</b>: 설명/태그/메타데이터
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 🏭 활용 분야")
        st.markdown("""
        - **디자인 검색**: 유사 부품 탐색
        - **품질 예측**: 제조 결함 사전 감지
        - **공정 최적화**: 파라미터 자동 추천
        - **지식 관리**: 설계 노하우 학습
        """)

        st.markdown("---")
        st.markdown("### ℹ️ 정보")
        st.markdown(f"버전: 1.0 | {datetime.now().strftime('%Y-%m-%d')}")

    # ============================================================
    # 모드별 콘텐츠
    # ============================================================

    if demo_mode == 'explore':
        render_explore_mode()
    elif demo_mode == 'embed':
        render_embedding_mode()
    elif demo_mode == 'search':
        render_search_mode()
    elif demo_mode == 'learn':
        render_learning_mode()


def render_explore_mode():
    """데이터 탐색 모드"""
    st.markdown("## 🗂️ Thingiverse 스타일 데이터 탐색")

    st.markdown("""
    <div class='info-card'>
    <b>Thingiverse란?</b> 세계 최대 3D 프린팅 디자인 공유 플랫폼.<br>
    200만 개 이상의 3D 모델과 메타데이터가 멀티모달 학습의 이상적인 데이터셋이 됩니다.
    </div>
    """, unsafe_allow_html=True)

    # 카테고리 필터
    categories = list(set(d['category'] for d in SAMPLE_DESIGNS))
    selected_cat = st.selectbox("카테고리 선택", ["전체"] + categories)

    filtered_designs = SAMPLE_DESIGNS if selected_cat == "전체" else [d for d in SAMPLE_DESIGNS if d['category'] == selected_cat]

    # 디자인 카드 표시
    cols = st.columns(2)
    for i, design in enumerate(filtered_designs):
        with cols[i % 2]:
            st.markdown(f"""
            <div class='modal-card'>
                <h4>{design['name']}</h4>
                <p><b>카테고리:</b> {design['category']}</p>
                <p>{design['description']}</p>
                <p>
                    <span class='similarity-high'>⬇️ {design['downloads']:,}</span>
                    <span class='similarity-med'>❤️ {design['likes']}</span>
                </p>
                <p><b>소재:</b> {design['material']}</p>
                <p><b>태그:</b> {', '.join(design['tags'])}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 데이터 통계
    st.markdown("### 📈 데이터셋 통계 (시뮬레이션)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 디자인 수", "2,100,000+", "+12% YoY")
    col2.metric("평균 다운로드", "8,500", "+5%")
    col3.metric("카테고리", "150+", "")
    col4.metric("일일 업로드", "5,000+", "+8%")

    # 멀티모달 데이터 구성
    st.markdown("### 🧩 멀티모달 데이터 구성")

    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=[go.Pie(
            labels=['3D 모델 (STL/OBJ)', '썸네일 이미지', '텍스트 설명', '메타데이터'],
            values=[35, 25, 20, 20],
            hole=.4,
            marker_colors=['#667eea', '#f5576c', '#11998e', '#f093fb']
        )])
        fig.update_layout(
            title_text="멀티모달 데이터 비율",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)


def render_embedding_mode():
    """임베딩 시연 모드"""
    st.markdown("## 🧬 멀티모달 임베딩 시연")

    st.markdown("""
    <div class='info-card'>
    <b>멀티모달 임베딩이란?</b><br>
    3D 형상, 이미지, 텍스트를 동일한 벡터 공간에 표현하여<br>
    서로 다른 모달리티 간 의미적 유사성을 비교할 수 있게 합니다.
    </div>
    """, unsafe_allow_html=True)

    # 3개 컬럼으로 입력
    col1, col2, col3 = st.columns(3)

    embeddings = {}

    with col1:
        st.markdown("### 📝 텍스트 입력")
        text_input = st.text_area(
            "디자인 설명",
            value="경량화된 항공 부품, 격자 구조, 티타늄 소재, 위상 최적화 적용",
            height=120
        )
        if text_input:
            text_emb = generate_mock_embedding(text_input)
            embeddings['text'] = text_emb
            st.markdown(f"""
            <div class='feature-box'>
                ✅ 텍스트 임베딩 생성<br>
                차원: 128 | Norm: {np.linalg.norm(text_emb):.4f}
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🖼️ 이미지 입력")
        image_file = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'], key='embed_img')

        if image_file and Image:
            img = Image.open(image_file)
            st.image(img, use_container_width=True)
            img_array = np.array(img.convert('RGB'))
            img_emb = generate_image_embedding(img_array)
            embeddings['image'] = img_emb
            st.markdown(f"""
            <div class='feature-box'>
                ✅ 이미지 임베딩 생성<br>
                차원: 128 | Norm: {np.linalg.norm(img_emb):.4f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("이미지를 업로드하세요")

    with col3:
        st.markdown("### 🎨 3D 모델")
        model_type = st.selectbox("샘플 모델 선택", ["cube", "gear", "lattice"])

        vertices, faces, _ = create_sample_3d_model(model_type)

        if PLOTLY_AVAILABLE:
            fig = visualize_3d_mesh(vertices, faces)
            st.plotly_chart(fig, use_container_width=True)

        model_emb = generate_3d_embedding(vertices)
        embeddings['3d'] = model_emb
        st.markdown(f"""
        <div class='feature-box'>
            ✅ 3D 임베딩 생성<br>
            정점: {len(vertices)} | Norm: {np.linalg.norm(model_emb):.4f}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 임베딩 융합 및 시각화
    if len(embeddings) >= 2:
        st.markdown("### 🔮 멀티모달 임베딩 융합")

        # 융합 임베딩 생성
        emb_list = list(embeddings.values())
        fused_emb = np.mean(emb_list, axis=0)
        fused_emb = fused_emb / np.linalg.norm(fused_emb)
        embeddings['fused'] = fused_emb

        col1, col2 = st.columns([2, 1])

        with col1:
            if PLOTLY_AVAILABLE:
                fig = visualize_embedding(embeddings)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 📊 모달리티 간 유사도")

            keys = [k for k in embeddings.keys() if k != 'fused']
            for i, k1 in enumerate(keys):
                for k2 in keys[i+1:]:
                    sim = cosine_similarity(embeddings[k1], embeddings[k2])
                    css_class = 'similarity-high' if sim > 0.5 else ('similarity-med' if sim > 0.2 else 'similarity-low')
                    st.markdown(f"**{k1} ↔ {k2}:** <span class='{css_class}'>{sim:.3f}</span>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 🎯 융합 임베딩 활용")
            st.markdown("""
            - 크로스모달 검색
            - 디자인 추천
            - 품질 예측
            - 지식 전이
            """)


def render_search_mode():
    """유사 검색 모드"""
    st.markdown("## 🔍 멀티모달 유사 디자인 검색")

    st.markdown("""
    <div class='info-card'>
    텍스트 설명, 이미지, 또는 3D 모델로 유사한 디자인을 검색합니다.<br>
    멀티모달 임베딩 공간에서 코사인 유사도 기반 검색을 수행합니다.
    </div>
    """, unsafe_allow_html=True)

    # 검색 모드 선택
    search_mode = st.radio(
        "검색 방식",
        ["텍스트 검색", "이미지 검색", "3D 모델 검색"],
        horizontal=True
    )

    query_embedding = None

    if search_mode == "텍스트 검색":
        query_text = st.text_input(
            "검색 쿼리",
            value="경량 항공 부품, 격자 구조",
            placeholder="검색할 디자인 특성을 입력하세요"
        )
        if query_text:
            query_embedding = generate_mock_embedding(query_text)
            st.success(f"쿼리 임베딩 생성 완료 (128차원)")

    elif search_mode == "이미지 검색":
        query_image = st.file_uploader("검색할 이미지", type=['png', 'jpg', 'jpeg'], key='search_img')
        if query_image and Image:
            col1, col2 = st.columns([1, 2])
            with col1:
                img = Image.open(query_image)
                st.image(img, caption="검색 이미지", use_container_width=True)
            with col2:
                img_array = np.array(img.convert('RGB'))
                query_embedding = generate_image_embedding(img_array)
                st.success("이미지 임베딩 생성 완료")

    else:  # 3D 모델 검색
        stl_file = st.file_uploader("STL 파일 업로드", type=['stl'], key='search_stl')

        if stl_file:
            try:
                content = stl_file.read().decode('utf-8')
                vertices, _ = parse_stl_ascii(content)
                if len(vertices) > 0:
                    query_embedding = generate_3d_embedding(vertices)
                    st.success(f"3D 임베딩 생성 완료 (정점: {len(vertices)}개)")

                    if PLOTLY_AVAILABLE:
                        fig = visualize_3d_mesh(vertices)
                        st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("바이너리 STL 또는 파싱 오류. 샘플 모델을 사용합니다.")
                vertices, faces, _ = create_sample_3d_model("gear")
                query_embedding = generate_3d_embedding(vertices)
        else:
            st.info("STL 파일을 업로드하거나 샘플을 사용하세요")
            if st.button("샘플 3D 모델로 검색"):
                vertices, faces, _ = create_sample_3d_model("lattice")
                query_embedding = generate_3d_embedding(vertices)

    # 검색 실행
    st.markdown("---")

    if query_embedding is not None:
        if st.button("🔍 유사 디자인 검색", use_container_width=True):
            with st.spinner("검색 중..."):
                results = search_similar_designs(query_embedding, SAMPLE_DESIGNS, top_k=3)
                st.session_state.search_results = results

    # 검색 결과 표시
    if st.session_state.search_results:
        st.markdown("### 🎯 검색 결과")

        for i, result in enumerate(st.session_state.search_results):
            sim = result['similarity']
            css_class = 'similarity-high' if sim > 0.5 else ('similarity-med' if sim > 0.2 else 'similarity-low')

            with st.expander(f"#{i+1} {result['name']} - 유사도: {sim:.3f}", expanded=(i==0)):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**카테고리:** {result['category']}")
                    st.markdown(f"**설명:** {result['description']}")
                    st.markdown(f"**태그:** {', '.join(result['tags'])}")
                    st.markdown(f"**소재:** {result['material']}")
                    st.markdown(f"**출력 시간:** {result['print_time']}")

                with col2:
                    st.metric("다운로드", f"{result['downloads']:,}")
                    st.metric("좋아요", result['likes'])
                    st.markdown(f"<span class='{css_class}'>유사도: {sim:.1%}</span>", unsafe_allow_html=True)


def render_learning_mode():
    """학습 시뮬레이션 모드"""
    st.markdown("## 📚 멀티모달 파운데이션 모델 학습 시뮬레이션")

    st.markdown("""
    <div class='info-card'>
    <b>파운데이션 모델 학습 파이프라인</b><br>
    Thingiverse 데이터로 3D + 이미지 + 텍스트를 동시에 학습하는 과정을 시뮬레이션합니다.
    </div>
    """, unsafe_allow_html=True)

    # 학습 구성
    st.markdown("### ⚙️ 학습 구성")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 데이터셋")
        data_size = st.slider("학습 데이터 수", 10000, 500000, 100000, step=10000)
        st.markdown(f"- 3D 모델: **{data_size:,}**개")
        st.markdown(f"- 이미지: **{data_size*2:,}**개")
        st.markdown(f"- 텍스트: **{data_size:,}**개")

    with col2:
        st.markdown("#### 모델 아키텍처")
        model_arch = st.selectbox("기반 모델", ["ViT-L/14", "ViT-H/14", "ViT-G/14"])
        emb_dim = st.selectbox("임베딩 차원", [256, 512, 768, 1024])
        st.markdown(f"- 파라미터: **{{'ViT-L/14': '428M', 'ViT-H/14': '632M', 'ViT-G/14': '1.8B'}[model_arch]}**")

    with col3:
        st.markdown("#### 학습 설정")
        epochs = st.slider("에폭 수", 1, 100, 30)
        batch_size = st.selectbox("배치 크기", [64, 128, 256, 512])
        lr = st.selectbox("학습률", ["1e-4", "5e-5", "1e-5"])

    st.markdown("---")

    # 학습 시뮬레이션
    st.markdown("### 🚀 학습 시뮬레이션")

    if st.button("학습 시작", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()

        # 시뮬레이션 학습 루프
        for epoch in range(1, min(epochs, 10) + 1):
            progress = epoch / min(epochs, 10)
            progress_bar.progress(progress)

            # 가상 메트릭
            train_loss = 2.5 * np.exp(-0.1 * epoch) + np.random.randn() * 0.05
            val_loss = 2.6 * np.exp(-0.1 * epoch) + np.random.randn() * 0.05
            text_3d_sim = 0.3 + 0.05 * epoch + np.random.randn() * 0.02
            img_3d_sim = 0.25 + 0.06 * epoch + np.random.randn() * 0.02

            status_text.markdown(f"**에폭 {epoch}/{min(epochs, 10)}** - 배치 처리 중...")

            with metrics_container.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Train Loss", f"{train_loss:.4f}")
                col2.metric("Val Loss", f"{val_loss:.4f}")
                col3.metric("Text-3D 유사도", f"{min(text_3d_sim, 0.95):.3f}")
                col4.metric("Image-3D 유사도", f"{min(img_3d_sim, 0.92):.3f}")

            import time
            time.sleep(0.5)

        st.success("✅ 학습 시뮬레이션 완료!")
        st.balloons()

    st.markdown("---")

    # 학습 결과 시각화 (가상)
    st.markdown("### 📊 예상 학습 결과")

    col1, col2 = st.columns(2)

    with col1:
        if PLOTLY_AVAILABLE:
            epochs_x = list(range(1, 31))
            train_losses = [2.5 * np.exp(-0.1 * e) for e in epochs_x]
            val_losses = [2.6 * np.exp(-0.1 * e) + 0.05 for e in epochs_x]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs_x, y=train_losses, name='Train Loss', mode='lines'))
            fig.add_trace(go.Scatter(x=epochs_x, y=val_losses, name='Val Loss', mode='lines'))
            fig.update_layout(title='손실 함수 곡선', xaxis_title='Epoch', yaxis_title='Loss', height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if PLOTLY_AVAILABLE:
            epochs_x = list(range(1, 31))
            text_3d = [min(0.3 + 0.025 * e, 0.95) for e in epochs_x]
            img_3d = [min(0.25 + 0.03 * e, 0.92) for e in epochs_x]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs_x, y=text_3d, name='Text-3D', mode='lines'))
            fig.add_trace(go.Scatter(x=epochs_x, y=img_3d, name='Image-3D', mode='lines'))
            fig.update_layout(title='크로스모달 유사도', xaxis_title='Epoch', yaxis_title='Similarity', height=300)
            st.plotly_chart(fig, use_container_width=True)

    # 활용 시나리오
    st.markdown("### 🎯 학습된 모델 활용 시나리오")

    use_cases = st.columns(3)

    with use_cases[0]:
        st.markdown("""
        <div class='modal-card'>
            <h4>🔍 디자인 검색</h4>
            <p>텍스트로 3D 모델 검색<br>
            "경량 격자 구조" → 관련 STL</p>
        </div>
        """, unsafe_allow_html=True)

    with use_cases[1]:
        st.markdown("""
        <div class='modal-card'>
            <h4>🏭 품질 예측</h4>
            <p>3D 형상에서 결함 예측<br>
            STL 분석 → 출력 성공률</p>
        </div>
        """, unsafe_allow_html=True)

    with use_cases[2]:
        st.markdown("""
        <div class='modal-card'>
            <h4>⚙️ 공정 최적화</h4>
            <p>형상 기반 파라미터 추천<br>
            STL → 레이저 출력/속도</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
