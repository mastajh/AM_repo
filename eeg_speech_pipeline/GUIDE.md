# EEG Speech Decoding Pipeline - 상세 가이드

> **목적**: 파이프라인의 설계 의도, 구현 결과물 분석, 그리고 실제 데이터 적용 시 반드시 고려해야 할 사항들을 정리한 문서

---

## 1. 작성 의도 (Design Intent)

### 1.1 왜 이 구조인가?

EEG-to-Text 변환은 단순한 분류 문제가 아니라 **다차원 신호에서 의미를 추출하는 문제**입니다. 기존 딥러닝 end-to-end 접근은 대규모 데이터가 필요하고, EEG 데이터는 본질적으로 소규모(피험자당 수백 trial)입니다.

이 파이프라인은 다음 전략을 채택합니다:

```
학습 최소화 + 수학적 특징 추출 + RAG 기반 생성
```

| 전략 | 이유 |
|------|------|
| Wavelet (학습 불필요) | 음소/주파수 정보는 신호 처리로 충분히 추출 가능 |
| Riemannian (캘리브레이션만) | 공분산 구조는 의도 상태를 잘 반영하며, 학습 파라미터가 참조 행렬 하나뿐 |
| 가중 결합 | 두 인코더의 차원 불균형(1920 vs 2080)을 수학적으로 보정 |
| 벡터 DB + RAG | 소규모 데이터에서도 유사 패턴 검색으로 확장 가능 |

### 1.2 각 모듈의 설계 근거

#### Preprocessor

```
의도: EEG 고유의 노이즈를 제거하면서 유용한 주파수 대역만 남기기
```

- **Bandpass 0.5-50Hz**: 뇌파의 유효 대역(delta~gamma)만 보존. DC drift와 고주파 근전도(EMG) 아티팩트 제거
- **Notch 60Hz**: 전원선 간섭 제거 (한국/미국 기준 60Hz, 유럽은 50Hz로 변경 필요)
- **Z-score 정규화**: 채널 간 진폭 차이 보정. 피험자/세션 간 변이 감소

설계 선택:
- `filtfilt` 사용 → 위상 왜곡 없음 (실시간 처리 시에는 `lfilter`로 변경 필요)
- 4차 Butterworth → 급격한 차단 없이 안정적 필터링
- 채널별 독립 정규화 → 채널 간 상대적 패턴 보존

#### Wavelet Encoder

```
의도: 시간-주파수 동시 분석으로 음소/조음 관련 특징 추출
```

- **db4 웨이블릿**: Daubechies-4는 EEG에서 가장 검증된 웨이블릿. 비대칭성이 뇌파 형태와 잘 맞음
- **5레벨 분해**: 512Hz 기준으로 각 레벨이 특정 뇌파 대역에 대응
  - Level 5 (cA5): 0-8Hz → delta+theta (깊은 인지 상태)
  - Level 4 (cD4): 8-16Hz → alpha (억제/이완)
  - Level 3 (cD3): 16-32Hz → beta (운동/언어 관련)
  - Level 2 (cD2): 32-64Hz → low gamma (인지 결합)
  - Level 1 (cD1): 64-128Hz → high gamma (노이즈 가능성)
- **5가지 통계 특징**: mean, std, rms, energy, entropy → 각 대역의 활성 정도를 다각도로 포착
- **채널별 concat**: 공간적 정보 보존 (평균하면 위치 정보 손실)

왜 학습 불필요한가:
- 웨이블릿 계수 자체가 이미 시간-주파수 분해의 결과
- 통계 특징은 수학적으로 정의되므로 데이터에 의존하지 않음
- 새로운 피험자에도 즉시 적용 가능

#### Riemannian Encoder

```
의도: 채널 간 상호작용(공분산) 구조에서 의도 상태를 추출
```

- **공분산 행렬**: 뇌 영역 간 동기화 패턴을 담음. 같은 의도를 가질 때 유사한 공분산 구조를 보임
- **리만 기하학**: 공분산 행렬은 양정치(SPD) 매니폴드 위의 점. 유클리드 거리보다 리만 거리가 더 의미있는 비교를 제공
- **탄젠트 공간 투영**: SPD 매니폴드를 참조점(리만 평균) 기준으로 평탄화하여 유클리드 연산 가능하게 함
- **상삼각 추출**: 대칭행렬이므로 절반만 저장. √2 스케일링으로 Frobenius norm 보존

왜 fit()이 필요한가:
- 탄젠트 공간 투영에 참조점(리만 평균)이 필요
- 이 참조점은 해당 피험자의 "평균적 뇌 상태"를 나타냄
- 피험자마다 뇌 구조/연결성이 다르므로 개별 캘리브레이션 필수

#### AutoWeightedCombiner

```
의도: 차원 수 차이에 의한 암묵적 편향을 제거하고 발언권을 명시적으로 제어
```

문제 상황:
- Wavelet: 1920차원, Riemannian: 2080차원
- 단순 concat 시 Riemannian이 코사인 유사도에서 더 많은 영향력 행사
- 이는 의도된 것이 아니라 차원 수의 부작용

해결 공식:
```
w_scale = √r        (Wavelet 발언권)
r_scale = √(1-r)    (Riemannian 발언권)

결합 벡터 = [w_scale × L2_norm(wav), r_scale × L2_norm(riem)]
```

수학적 보장:
- L2 정규화로 각 인코더 출력의 크기를 1로 통일
- √r 스케일링으로 코사인 유사도에서 r 비율만큼 기여하도록 보장
- `cos_sim(a,b) = r × wav_sim(a,b) + (1-r) × riem_sim(a,b)` 가 성립

#### Vector Database

```
의도: 소규모 데이터에서 유사 패턴을 효율적으로 검색
```

- **FAISS**: 대규모 벡터 검색에 최적화된 라이브러리. CPU에서도 빠름
- **코사인 유사도**: 벡터 크기가 아닌 방향으로 비교. 정규화 후 내적(IP)으로 구현
- **메타데이터 저장**: 각 벡터에 의도 레이블과 후보 텍스트를 연결

#### LLM Generator

```
의도: 검색된 유사 의도들을 자연어로 합성/정제
```

- **RAG 방식**: 벡터 DB 검색 결과를 LLM에게 제공하여 문맥에 맞는 문장 생성
- **더미 모드**: OpenAI API 없이도 동작 가능 (가장 유사한 결과의 텍스트 반환)
- **온도 0.7**: 다양성과 정확성의 균형

### 1.3 설계 트레이드오프

| 결정 | 장점 | 단점 | 대안 |
|------|------|------|------|
| 채널별 concat (Wavelet) | 공간 정보 보존 | 차원 수 큼 (1920) | 채널 평균 (30dim) |
| 리만 평균 (50회 반복) | 정확한 참조점 | 계산 비용 | LogEuclid 평균 (1회) |
| FAISS flat index | 정확한 검색 | 대규모에서 느림 | IVF/HNSW (근사 검색) |
| 단일 LLM 호출 | 간단 | API 의존성 | 로컬 모델 (Llama 등) |

---

## 2. 결과물 분석 (Implementation Results)

### 2.1 데이터 흐름과 차원 변화

```
입력: Raw EEG
  Shape: (64, 512)  →  64채널, 1초(512Hz)

    ↓ Preprocessor
  Shape: (64, 512)  →  동일 크기, 필터링+정규화됨

    ↓ Wavelet Encoder
  Shape: (1920,)    →  64ch × 6levels × 5features = 1920

    ↓ Riemannian Encoder
  Shape: (2080,)    →  64×65/2 = 2080 (상삼각)

    ↓ AutoWeightedCombiner (r=0.5)
  Shape: (4000,)    →  1920 + 2080 = 4000
  스케일: [√0.5 × norm(wav), √0.5 × norm(riem)]

    ↓ Vector DB Search
  결과: [(0.85, {intent, texts}), (0.72, {...}), ...]

    ↓ LLM Generator
  출력: "물 주세요"
```

### 2.2 모듈별 출력 검증

#### Preprocessor 출력 특성
- 채널별 평균 ≈ 0 (Z-score)
- 채널별 표준편차 ≈ 1 (Z-score)
- 주파수 범위: 0.5-50Hz만 포함
- 60Hz 성분 제거됨

#### Wavelet 임베딩 특성
- 결정론적 (같은 입력 → 같은 출력)
- 값 범위: 특징에 따라 다름 (energy는 양수, mean은 실수)
- 채널 순서에 의존 (채널 순서 변경 시 결과 변경)

#### Riemannian 임베딩 특성
- 참조 행렬에 의존 (다른 캘리브레이션 → 다른 출력)
- 실수값 (허수부 제거됨)
- 양정치 행렬의 로그 맵 결과이므로 대칭

#### Combined 임베딩 특성
- 전체 L2 norm ≈ 1 (r=0.5일 때 정확히 1)
- Wavelet 부분 L2 norm = √r
- Riemannian 부분 L2 norm = √(1-r)

### 2.3 테스트 커버리지

현재 26개 테스트가 다음을 검증합니다:

| 카테고리 | 테스트 수 | 검증 내용 |
|----------|-----------|-----------|
| Preprocessor | 4 | 초기화, 출력 형태, 정규화, 배치 처리 |
| WaveletEncoder | 5 | 초기화, 출력 형태, 차원 계산, 결정론성, 배치 |
| RiemannianEncoder | 5 | 초기화, fit, 미fit 에러, 출력 형태, 실수값 |
| AutoWeightedCombiner | 7 | 초기화, 스케일, 형태, 정규화, 비율 변경, 배치, 분석 |
| Integration | 3 | 전처리→Wavelet, 전처리→Riemannian, 전체 인코딩 |

---

## 3. 실제 데이터 적용 시 고려사항

### 3.1 데이터 획득 및 준비

#### Task 1: Inner Speech Dataset 다운로드

```bash
# OpenNeuro에서 다운로드 (ds003626)
# 방법 1: openneuro-py
pip install openneuro-py
openneuro-py download --dataset ds003626 --target-dir data/raw/

# 방법 2: AWS CLI
aws s3 sync --no-sign-request \
  s3://openneuro.org/ds003626 data/raw/ds003626/
```

**주의사항**:
- 전체 용량: ~50GB (10명 피험자)
- 128채널 → 64채널로 다운샘플링 필요 (또는 config 수정)
- 1024Hz → 512Hz 리샘플링 필요 (또는 config 수정)

#### Task 2: 채널 수 조정

현재 파이프라인은 **64채널, 512Hz** 기준으로 설계되었습니다.

```python
# Inner Speech Dataset은 128채널, 1024Hz
# 옵션 1: config.yaml에서 n_channels: 128로 변경
# → Wavelet: 128 × 30 = 3840dim
# → Riemannian: 128 × 129 / 2 = 8256dim
# → 총: 12096dim (메모리/속도 이슈 가능)

# 옵션 2: 채널 서브셋 선택 (권장)
# 10-20 시스템 기준 64채널 선택
selected_channels = [...]  # 64개 채널 인덱스
eeg_subset = eeg_raw[selected_channels, :]

# 옵션 3: 리샘플링
from scipy.signal import resample
eeg_resampled = resample(eeg_raw, num=512, axis=1)  # 1초 기준
```

#### Task 3: 에폭 추출

```python
# Inner Speech Dataset 구조:
# - 각 trial에 이벤트 마커 존재
# - "inner speech" 구간만 추출해야 함

import mne

raw = mne.io.read_raw_bdf('sub-01_task-innerspeech_eeg.bdf')
events = mne.find_events(raw)

# 이벤트 코드 매핑 (데이터셋 문서 참조)
event_id = {
    'up': 1,
    'down': 2,
    'left': 3,
    'right': 4
}

epochs = mne.Epochs(raw, events, event_id,
                    tmin=0, tmax=1.0,  # 1초 에폭
                    baseline=None,
                    preload=True)

# numpy 변환
eeg_data = epochs.get_data()  # (n_trials, n_channels, n_timepoints)
labels = epochs.events[:, -1]
```

### 3.2 캘리브레이션 전략

#### Task 4: 적절한 캘리브레이션 데이터 선택

```python
# 권장: 모든 클래스에서 균등하게 샘플링
calibration_indices = []
for label in unique_labels:
    class_indices = np.where(labels == label)[0]
    # 각 클래스에서 12-15개씩 (총 50-60개)
    selected = np.random.choice(class_indices, size=min(15, len(class_indices)), replace=False)
    calibration_indices.extend(selected)

calibration_data = eeg_data[calibration_indices]
pipeline.calibrate(calibration_data)
```

**주의사항**:
- 최소 20개, 권장 50개 이상의 샘플
- 모든 의도 클래스가 포함되어야 함 (편향 방지)
- 캘리브레이션 데이터는 DB 구축에도 사용 가능 (중복 OK)
- 세션이 바뀌면 재캘리브레이션 고려 (뇌 상태 변화)

#### Task 5: 캘리브레이션 품질 검증

```python
# 참조 행렬이 양정치인지 확인
eigenvalues = np.linalg.eigvalsh(pipeline.riemannian_encoder.reference_)
assert np.all(eigenvalues > 0), "참조 행렬이 양정치가 아닙니다!"
print(f"최소 고유값: {eigenvalues.min():.6f}")
print(f"조건수: {eigenvalues.max() / eigenvalues.min():.1f}")
# 조건수가 10000 이상이면 정규화 강도(epsilon) 증가 필요
```

### 3.3 DB 구축

#### Task 6: 메타데이터 구성

```python
# 각 EEG 샘플에 대한 메타데이터 준비
metadata_list = []
for i, label in enumerate(labels):
    # 의도별 후보 문장 매핑
    intent_to_texts = {
        'up': ['위로 가세요', '위쪽입니다', '올려주세요'],
        'down': ['아래로 가세요', '아래쪽입니다', '내려주세요'],
        'left': ['왼쪽으로 가세요', '왼쪽입니다', '왼쪽이요'],
        'right': ['오른쪽으로 가세요', '오른쪽입니다', '오른쪽이요'],
    }

    intent = label_to_intent[label]  # 숫자→문자 매핑
    metadata_list.append({
        'intent': intent,
        'candidate_texts': intent_to_texts[intent],
        'subject_id': 'sub-01',
        'session': 'ses-01',
        'trial_idx': i,
    })
```

#### Task 7: 피험자별 DB 분리

```python
# 피험자별로 독립적인 파이프라인/DB 구축
for subject_id in subject_ids:
    # 해당 피험자 데이터 로드
    subject_eeg = load_subject_data(subject_id)
    subject_labels = load_subject_labels(subject_id)

    # 독립 파이프라인 생성
    pipeline = EEGSpeechPipeline.from_config('config/config.yaml')

    # 캘리브레이션
    pipeline.calibrate(subject_eeg[:50])

    # DB 구축
    pipeline.build_database(subject_eeg, metadata_list)

    # 저장
    pipeline.save(f'data/pipeline/{subject_id}')
```

**절대 하지 말아야 할 것**:
- 여러 피험자 데이터를 하나의 DB에 혼합 → 검색 정확도 급감
- 피험자 A의 캘리브레이션으로 피험자 B 인코딩 → 탄젠트 공간 불일치

#### Task 8: DB 크기와 검색 성능 검토

```python
# DB 크기에 따른 인덱스 타입 선택
n_vectors = len(eeg_data)

if n_vectors < 10000:
    index_type = 'flat'  # 정확한 검색, 충분히 빠름
elif n_vectors < 100000:
    index_type = 'ivf'   # 근사 검색, nlist=100
else:
    index_type = 'hnsw'  # 대규모, M=32

# config.yaml에 반영
# vector_db:
#   index_type: flat  # 또는 ivf, hnsw
```

### 3.4 발언권 비율(r) 튜닝

#### Task 9: 최적 r 값 탐색

```python
# 교차 검증으로 최적 r 탐색
from sklearn.model_selection import StratifiedKFold

results = {}
for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    accuracies = []

    skf = StratifiedKFold(n_splits=5)
    for train_idx, test_idx in skf.split(eeg_data, labels):
        # 파이프라인 생성
        pipeline = EEGSpeechPipeline.from_config('config/config.yaml')
        pipeline.combiner.set_ratio(r)

        # 캘리브레이션 + DB 구축
        pipeline.calibrate(eeg_data[train_idx[:50]])
        pipeline.build_database(eeg_data[train_idx], [metadata_list[i] for i in train_idx])

        # 평가
        correct = 0
        for idx in test_idx:
            result = pipeline.predict_with_details(eeg_data[idx])
            if result['top_intent'] == label_to_intent[labels[idx]]:
                correct += 1

        accuracies.append(correct / len(test_idx))

    results[r] = np.mean(accuracies)
    print(f"r={r:.1f}: accuracy={np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")

best_r = max(results, key=results.get)
print(f"\n최적 r: {best_r}")
```

**r 값 해석 가이드**:
- `r > 0.5` 최적 → Wavelet(시간-주파수) 정보가 더 구분력 있음 → 주파수 대역 특징이 중요한 태스크
- `r < 0.5` 최적 → Riemannian(공분산) 정보가 더 구분력 있음 → 채널 간 연결성이 중요한 태스크
- `r ≈ 0.5` 최적 → 두 정보가 상보적으로 작용 → 이상적인 상황

### 3.5 성능 평가

#### Task 10: 분류 정확도 평가 (의도 수준)

```python
# 벡터 DB 검색 top-1의 의도가 정답과 일치하는지 평가
def evaluate_intent_accuracy(pipeline, test_eeg, test_labels, label_map):
    correct = 0
    confusion = np.zeros((len(label_map), len(label_map)), dtype=int)

    for eeg, label in zip(test_eeg, test_labels):
        result = pipeline.predict_with_details(eeg, top_k=5)
        predicted_intent = result['top_intent']
        true_intent = label_map[label]

        pred_idx = list(label_map.values()).index(predicted_intent)
        true_idx = list(label_map.values()).index(true_intent)
        confusion[true_idx, pred_idx] += 1

        if predicted_intent == true_intent:
            correct += 1

    accuracy = correct / len(test_labels)
    return accuracy, confusion
```

#### Task 11: 검색 품질 평가

```python
# top-k 내에 정답이 포함되는 비율 (Recall@k)
def evaluate_recall_at_k(pipeline, test_eeg, test_labels, label_map, k_values=[1, 3, 5]):
    recalls = {k: 0 for k in k_values}

    for eeg, label in zip(test_eeg, test_labels):
        true_intent = label_map[label]
        result = pipeline.predict_with_details(eeg, top_k=max(k_values))

        for k in k_values:
            top_k_intents = [r[1]['intent'] for r in result['search_results'][:k]]
            if true_intent in top_k_intents:
                recalls[k] += 1

    for k in k_values:
        recalls[k] /= len(test_labels)
        print(f"Recall@{k}: {recalls[k]:.3f}")

    return recalls
```

#### Task 12: 유사도 분포 분석

```python
# 같은 클래스 vs 다른 클래스 간 유사도 분포 확인
def analyze_similarity_distribution(pipeline, eeg_data, labels):
    embeddings = pipeline.encode_batch(eeg_data)

    same_class_sims = []
    diff_class_sims = []

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
            )
            if labels[i] == labels[j]:
                same_class_sims.append(sim)
            else:
                diff_class_sims.append(sim)

    print(f"같은 클래스 유사도: {np.mean(same_class_sims):.3f} ± {np.std(same_class_sims):.3f}")
    print(f"다른 클래스 유사도: {np.mean(diff_class_sims):.3f} ± {np.std(diff_class_sims):.3f}")
    print(f"분리도 (gap): {np.mean(same_class_sims) - np.mean(diff_class_sims):.3f}")

    # 분리도가 0.1 미만이면 인코딩 품질 의심
```

### 3.6 실시간 적용 시 추가 고려사항

#### Task 13: 실시간 필터링 전환

```python
# filtfilt → lfilter로 변경 (인과적 필터)
# preprocessor.py 수정 필요

class RealtimePreprocessor(EEGPreprocessor):
    """실시간용 전처리기 (인과적 필터)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 필터 상태 초기화
        self.zi_bp = None
        self.zi_notch = None

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        실시간 청크 처리

        Args:
            chunk: (n_channels, chunk_size) - 보통 32~128 샘플
        """
        from scipy.signal import lfilter, lfilter_zi

        if self.zi_bp is None:
            # 초기 상태 계산
            zi = lfilter_zi(self.b_bp, self.a_bp)
            self.zi_bp = np.outer(np.ones(chunk.shape[0]), zi)
            zi = lfilter_zi(self.b_notch, self.a_notch)
            self.zi_notch = np.outer(np.ones(chunk.shape[0]), zi)

        # 필터링 (상태 유지)
        filtered, self.zi_bp = lfilter(self.b_bp, self.a_bp, chunk, axis=1, zi=self.zi_bp)
        filtered, self.zi_notch = lfilter(self.b_notch, self.a_notch, filtered, axis=1, zi=self.zi_notch)

        return filtered
```

#### Task 14: 추론 지연 시간 측정

```python
import time

# 각 단계별 지연 시간 측정
eeg_sample = np.random.randn(64, 512)

t0 = time.time()
processed = pipeline.preprocessor.process(eeg_sample)
t1 = time.time()
wav_emb = pipeline.wavelet_encoder.encode(processed)
t2 = time.time()
riem_emb = pipeline.riemannian_encoder.encode(processed)
t3 = time.time()
combined = pipeline.combiner.combine(wav_emb, riem_emb)
t4 = time.time()
results = pipeline.vector_db.search(combined, top_k=5)
t5 = time.time()

print(f"전처리:     {(t1-t0)*1000:.1f} ms")
print(f"Wavelet:    {(t2-t1)*1000:.1f} ms")
print(f"Riemannian: {(t3-t2)*1000:.1f} ms")  # 가장 느림 (행렬 연산)
print(f"결합:       {(t4-t3)*1000:.1f} ms")
print(f"검색:       {(t5-t4)*1000:.1f} ms")
print(f"총 (LLM 제외): {(t5-t0)*1000:.1f} ms")

# 목표: 전체 100ms 이하 (LLM 제외)
# Riemannian이 느리면 n_channels 줄이기 고려
```

#### Task 15: 버퍼링 전략

```python
# 실시간 EEG 스트림에서 1초 윈도우를 어떻게 구성할지

class EEGBuffer:
    """슬라이딩 윈도우 버퍼"""

    def __init__(self, n_channels=64, window_size=512, step_size=128):
        self.buffer = np.zeros((n_channels, window_size))
        self.window_size = window_size
        self.step_size = step_size  # 250ms마다 예측
        self.count = 0

    def add_chunk(self, chunk):
        """새 데이터 추가, 예측 가능하면 True 반환"""
        chunk_size = chunk.shape[1]

        # 버퍼 시프트
        self.buffer = np.roll(self.buffer, -chunk_size, axis=1)
        self.buffer[:, -chunk_size:] = chunk

        self.count += chunk_size
        return self.count >= self.step_size

    def get_window(self):
        """현재 윈도우 반환"""
        self.count = 0
        return self.buffer.copy()
```

### 3.7 아티팩트 처리

#### Task 16: 눈깜빡임/근전도 제거

```python
# ICA 기반 아티팩트 제거 (오프라인)
import mne

def remove_artifacts_ica(raw, n_components=20):
    """ICA로 눈깜빡임 성분 제거"""
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
    ica.fit(raw)

    # EOG 채널과 상관 높은 성분 찾기
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_indices

    raw_clean = ica.apply(raw.copy())
    return raw_clean

# 실시간에서는 ICA 대신 임계값 기반 거부
def reject_bad_epoch(eeg, threshold=100e-6):
    """진폭 기준 에폭 거부"""
    if np.max(np.abs(eeg)) > threshold:
        return True  # 거부
    return False  # 사용 가능
```

#### Task 17: 채널 불량 탐지

```python
def detect_bad_channels(eeg, z_threshold=3.0):
    """불량 채널 탐지 및 보간"""
    channel_vars = np.var(eeg, axis=1)
    z_scores = (channel_vars - np.mean(channel_vars)) / np.std(channel_vars)

    bad_channels = np.where(np.abs(z_scores) > z_threshold)[0]

    if len(bad_channels) > 0:
        # 인접 채널 평균으로 보간 (간단한 방법)
        for ch in bad_channels:
            neighbors = get_neighbor_channels(ch)  # 채널 위치 기반
            eeg[ch] = np.mean(eeg[neighbors], axis=0)

    return eeg, bad_channels
```

### 3.8 확장 가능성

#### Task 18: 더 풍부한 메타데이터 설계

현재 4가지 방향 명령어만 지원하지만, 실제 BCI 응용에서는 더 다양한 의도가 필요합니다:

```python
# 확장된 의도-텍스트 매핑 예시
extended_intents = {
    'pain': {
        'texts': ['아파요', '통증이 있어요', '여기가 아파요'],
        'category': 'physical',
        'urgency': 'high'
    },
    'thirst': {
        'texts': ['물 주세요', '목이 말라요', '음료 주세요'],
        'category': 'need',
        'urgency': 'medium'
    },
    'fatigue': {
        'texts': ['쉬고 싶어요', '피곤해요', '잠깐 쉴게요'],
        'category': 'physical',
        'urgency': 'low'
    },
    'yes': {
        'texts': ['네', '맞아요', '좋아요'],
        'category': 'response',
        'urgency': 'low'
    },
    'no': {
        'texts': ['아니요', '싫어요', '그건 아니에요'],
        'category': 'response',
        'urgency': 'low'
    }
}
```

#### Task 19: 온라인 학습 (DB 점진적 업데이트)

```python
def update_database_online(pipeline, new_eeg, confirmed_intent, confirmed_text):
    """사용자 피드백 기반 DB 업데이트"""

    # 새 샘플 인코딩
    embedding = pipeline.encode(new_eeg)

    # 메타데이터 구성
    new_metadata = {
        'intent': confirmed_intent,
        'candidate_texts': [confirmed_text],
        'source': 'online_feedback',
        'timestamp': datetime.now().isoformat()
    }

    # DB에 추가
    pipeline.vector_db.add(
        embedding.reshape(1, -1),
        [new_metadata]
    )

    # 주기적으로 저장
    if len(pipeline.vector_db) % 100 == 0:
        pipeline.save('data/pipeline/updated')
```

#### Task 20: 다중 모달 확장

```python
# EMG, EOG 등 추가 생체 신호 통합 시
class MultiModalCombiner(AutoWeightedCombiner):
    """다중 모달 결합기"""

    def __init__(self, modality_dims: dict, ratios: dict):
        """
        Args:
            modality_dims: {'eeg_wav': 1920, 'eeg_riem': 2080, 'emg': 256}
            ratios: {'eeg_wav': 0.4, 'eeg_riem': 0.4, 'emg': 0.2}
        """
        self.modality_dims = modality_dims
        self.ratios = ratios
        self.scales = {k: np.sqrt(v) for k, v in ratios.items()}

    def combine(self, embeddings: dict) -> np.ndarray:
        parts = []
        for modality, emb in embeddings.items():
            normalized = emb / (np.linalg.norm(emb) + 1e-10)
            parts.append(self.scales[modality] * normalized)
        return np.concatenate(parts)
```

### 3.9 알려진 제한사항과 대응

| 제한사항 | 영향 | 대응 방안 |
|----------|------|-----------|
| Riemannian의 행렬 연산 비용 | 64ch에서 64×64 행렬 logm/expm 계산 | 채널 수 줄이기 (32ch), 또는 LogEuclid 근사 사용 |
| FAISS flat의 선형 검색 | DB가 10만개 이상이면 느려짐 | IVF 또는 HNSW로 전환 |
| LLM API 의존성 | 네트워크 지연, 비용 | 로컬 모델(Llama, KoGPT) 또는 더미 모드 |
| 4방향만 지원 | 실용성 부족 | 더 많은 의도 클래스 학습/DB 구축 |
| 피험자 간 전이 불가 | 새 피험자마다 캘리브레이션 필요 | Transfer learning 연구 또는 subject-independent 모델 |
| epoch 길이 고정 (1초) | 짧은/긴 의도 포착 불가 | 가변 길이 윈도우 또는 다중 스케일 분석 |

### 3.10 체크리스트: 실제 적용 전 확인 사항

```
□ 데이터 형식 확인 (채널 수, 샘플링 레이트, 에폭 길이)
□ config.yaml 파라미터 업데이트
□ 채널 선택/리샘플링 전처리 추가
□ 아티팩트 제거 파이프라인 구축 (ICA 또는 임계값)
□ 캘리브레이션 데이터 선정 (최소 20개, 클래스 균등)
□ 캘리브레이션 품질 검증 (참조 행렬 조건수)
□ DB 구축 후 유사도 분포 확인 (same vs diff class gap)
□ r 값 교차검증으로 최적화
□ top-k 검색 정확도 평가
□ 추론 지연 시간 측정 (목표치 이하 확인)
□ 실시간 적용 시 인과적 필터로 전환
□ 피험자별 독립 DB/파이프라인 구축
□ 에러 핸들링 (불량 채널, 아티팩트 에폭, API 실패)
□ 저장/로드 기능 검증
□ LLM 미사용 시 더미 모드 동작 확인
```

---

## 부록: 자주 묻는 질문

### Q: r=0.5인데 왜 기여도가 50:50이 아닌가요?

**A**: `r`은 "발언권"이지 "기여도"가 아닙니다. 실제 기여도는 `발언권 × 유사도`입니다.

```
예: r=0.5, wav_sim=0.9, riem_sim=0.3
→ wav 기여 = 0.5 × 0.9 = 0.45 (75%)
→ riem 기여 = 0.5 × 0.3 = 0.15 (25%)
```

이는 정상입니다. 더 "확신이 높은" (유사도가 높은) 정보가 자연스럽게 더 많이 기여합니다.

### Q: Wavelet 차원이 너무 큰데 줄일 수 있나요?

**A**: 세 가지 방법이 있습니다:

1. 채널 평균 사용 (30dim): `np.mean(channel_features, axis=0)` - 공간 정보 손실
2. 채널 수 줄이기: 64→32채널 → 960dim
3. 특징 수 줄이기: 5→3개 (mean, std, energy) → 64×6×3 = 1152dim

### Q: 128채널을 그대로 쓰면 Riemannian 차원이 8256인데 괜찮나요?

**A**: FAISS는 고차원도 처리 가능하지만, 128채널 Riemannian의 실제 문제는:
- `logm(128×128)` 계산이 느림 (~50ms)
- 8256차원 중 유효 차원이 실제로 적을 수 있음 (curse of dimensionality)

권장: PCA로 300-500차원으로 축소하거나, 채널 서브셋(32-64ch) 사용

### Q: FAISS 없이도 동작하나요?

**A**: 현재 구현은 FAISS 필수입니다. 대안:
- `faiss-cpu` 설치가 안 되면: numpy 기반 brute-force 검색 구현 가능 (느리지만 동작)
- 소규모 DB(<1000개)에서는 `sklearn.metrics.pairwise.cosine_similarity`로 충분

### Q: OpenAI API 없이 사용하려면?

**A**: LLMGenerator는 API 없이 더미 모드로 동작합니다:
- 검색 결과 중 가장 유사도 높은 항목의 첫 번째 `candidate_text`를 그대로 반환
- 이것만으로도 의도 분류→문장 매핑은 가능 (단, 유연한 문장 생성 불가)
