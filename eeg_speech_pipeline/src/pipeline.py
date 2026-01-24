"""EEG Speech Pipeline - full end-to-end pipeline orchestrator."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

from .preprocessor import EEGPreprocessor
from .encoders.wavelet_encoder import WaveletEncoder
from .encoders.riemannian_encoder import RiemannianEncoder
from .combiner import AutoWeightedCombiner
from .vector_db import VectorDatabase
from .llm_generator import LLMGenerator


class EEGSpeechPipeline:
    """
    EEG → Speech 전체 파이프라인

    사용법:
        # 1. 초기화
        pipeline = EEGSpeechPipeline.from_config('config/config.yaml')

        # 2. 캘리브레이션 (Riemannian용)
        pipeline.calibrate(calibration_eeg_samples)

        # 3. DB 구축
        pipeline.build_database(train_eeg, train_metadata)

        # 4. 추론
        text = pipeline.predict(new_eeg)
    """

    def __init__(
        self,
        preprocessor: EEGPreprocessor,
        wavelet_encoder: WaveletEncoder,
        riemannian_encoder: RiemannianEncoder,
        combiner: AutoWeightedCombiner,
        vector_db: VectorDatabase,
        llm_generator: LLMGenerator,
        config: Dict[str, Any],
    ):
        self.preprocessor = preprocessor
        self.wavelet_encoder = wavelet_encoder
        self.riemannian_encoder = riemannian_encoder
        self.combiner = combiner
        self.vector_db = vector_db
        self.llm_generator = llm_generator
        self.config = config

        self._is_calibrated = False

    @classmethod
    def from_config(cls, config_path: str) -> 'EEGSpeechPipeline':
        """설정 파일에서 파이프라인 생성"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        preprocessor = EEGPreprocessor(
            sampling_rate=config['preprocessor']['sampling_rate'],
            lowcut=config['preprocessor']['bandpass'][0],
            highcut=config['preprocessor']['bandpass'][1],
            notch_freq=config['preprocessor']['notch'],
        )

        wavelet_encoder = WaveletEncoder(
            wavelet=config['encoders']['wavelet']['wavelet'],
            level=config['encoders']['wavelet']['level'],
            features=config['encoders']['wavelet']['features'],
        )

        riemannian_encoder = RiemannianEncoder(
            metric=config['encoders']['riemannian']['metric'],
            n_iter=config['encoders']['riemannian'].get('n_iter', 50),
        )

        n_channels = config['data']['n_channels']
        wav_dim = wavelet_encoder.get_output_dim(n_channels)
        riem_dim = riemannian_encoder.get_output_dim(n_channels)
        combined_dim = wav_dim + riem_dim

        combiner = AutoWeightedCombiner(
            wav_dim=wav_dim,
            riem_dim=riem_dim,
            target_wav_ratio=config['combiner']['target_wav_ratio'],
        )

        vector_db = VectorDatabase(
            dim=combined_dim,
            index_type=config['vector_db']['index_type'],
            metric=config['vector_db']['metric'],
        )

        llm_generator = LLMGenerator(
            model=config['llm']['model'],
            temperature=config['llm']['temperature'],
        )

        return cls(
            preprocessor=preprocessor,
            wavelet_encoder=wavelet_encoder,
            riemannian_encoder=riemannian_encoder,
            combiner=combiner,
            vector_db=vector_db,
            llm_generator=llm_generator,
            config=config,
        )

    def calibrate(self, eeg_samples: np.ndarray) -> 'EEGSpeechPipeline':
        """
        Riemannian 인코더 캘리브레이션

        Args:
            eeg_samples: 캘리브레이션 EEG (n_samples, n_channels, n_timepoints)
        """
        print(f"캘리브레이션 시작: {len(eeg_samples)} 샘플")

        processed = self.preprocessor.process_batch(eeg_samples)
        self.riemannian_encoder.fit(processed)

        self._is_calibrated = True
        print("캘리브레이션 완료")

        return self

    def encode(self, eeg: np.ndarray) -> np.ndarray:
        """
        단일 EEG 샘플 인코딩

        Args:
            eeg: 원본 EEG (n_channels, n_timepoints)

        Returns:
            embedding: 결합된 임베딩 벡터
        """
        if not self._is_calibrated:
            raise RuntimeError("calibrate()를 먼저 호출하세요.")

        processed = self.preprocessor.process(eeg)
        wav_emb = self.wavelet_encoder.encode(processed)
        riem_emb = self.riemannian_encoder.encode(processed)
        combined = self.combiner.combine(wav_emb, riem_emb)

        return combined

    def encode_batch(self, eeg_batch: np.ndarray) -> np.ndarray:
        """배치 인코딩"""
        return np.array([self.encode(eeg) for eeg in eeg_batch])

    def build_database(
        self,
        eeg_samples: np.ndarray,
        metadata_list: List[Dict[str, Any]],
    ):
        """
        벡터 DB 구축

        Args:
            eeg_samples: EEG 샘플들 (n_samples, n_channels, n_timepoints)
            metadata_list: 각 샘플의 메타데이터
                [{'intent': '음료요청', 'candidate_texts': ['물 주세요', ...]}, ...]
        """
        print(f"DB 구축 시작: {len(eeg_samples)} 샘플")

        embeddings = self.encode_batch(eeg_samples)
        self.vector_db.add(embeddings, metadata_list)

        print(f"DB 구축 완료: {len(self.vector_db)} 벡터")

    def predict(
        self,
        eeg: np.ndarray,
        top_k: int = 5,
        context: Optional[str] = None,
    ) -> str:
        """
        EEG → 텍스트 변환

        Args:
            eeg: 입력 EEG (n_channels, n_timepoints)
            top_k: 검색할 유사 샘플 수
            context: 추가 상황 정보

        Returns:
            text: 생성된 문장
        """
        query = self.encode(eeg)
        results = self.vector_db.search(query, top_k=top_k)
        text = self.llm_generator.generate(results, context=context)

        return text

    def predict_with_details(
        self,
        eeg: np.ndarray,
        top_k: int = 5,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """상세 정보와 함께 예측"""
        query = self.encode(eeg)
        results = self.vector_db.search(query, top_k=top_k)
        text = self.llm_generator.generate(results, context=context)

        return {
            'text': text,
            'search_results': results,
            'embedding': query,
            'top_intent': results[0][1]['intent'] if results else None,
            'top_similarity': results[0][0] if results else None,
        }

    def save(self, path: str):
        """파이프라인 저장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.vector_db.save(str(path / 'vector_db'))
        self.riemannian_encoder.save(str(path / 'riemannian_reference.npy'))

        with open(path / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        print(f"파이프라인 저장: {path}")

    def load(self, path: str):
        """파이프라인 로드"""
        path = Path(path)

        self.vector_db.load(str(path / 'vector_db'))
        self.riemannian_encoder.load(str(path / 'riemannian_reference.npy'))

        self._is_calibrated = True
        print(f"파이프라인 로드: {path}")
