"""FAISS-based vector database for EEG embeddings."""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class VectorDatabase:
    """
    EEG 임베딩 벡터 데이터베이스

    FAISS 기반 코사인 유사도 검색

    Args:
        dim: 임베딩 차원
        index_type: 인덱스 타입 ('flat', 'ivf', 'hnsw')
        metric: 거리 메트릭 ('cosine', 'l2')
    """

    def __init__(
        self,
        dim: int,
        index_type: str = 'flat',
        metric: str = 'cosine',
    ):
        self.dim = dim
        self.index_type = index_type
        self.metric = metric
        self.metadata: List[Dict[str, Any]] = []

        if not HAS_FAISS:
            raise ImportError(
                "faiss-cpu 라이브러리가 필요합니다. "
                "pip install faiss-cpu 로 설치하세요."
            )

        self._create_index()

    def _create_index(self):
        """FAISS 인덱스 생성"""
        if self.metric == 'cosine':
            if self.index_type == 'flat':
                self.index = faiss.IndexFlatIP(self.dim)
            elif self.index_type == 'ivf':
                quantizer = faiss.IndexFlatIP(self.dim)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dim, 100, faiss.METRIC_INNER_PRODUCT
                )
            elif self.index_type == 'hnsw':
                self.index = faiss.IndexHNSWFlat(
                    self.dim, 32, faiss.METRIC_INNER_PRODUCT
                )
        else:
            if self.index_type == 'flat':
                self.index = faiss.IndexFlatL2(self.dim)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 정규화"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        return vectors / norms

    def add(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]],
    ):
        """
        벡터 추가

        Args:
            embeddings: (n, dim)
            metadata_list: 각 벡터의 메타데이터 리스트
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("임베딩 수와 메타데이터 수가 일치해야 합니다.")

        embeddings = embeddings.astype(np.float32)

        if self.metric == 'cosine':
            embeddings = self._normalize(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        유사 벡터 검색

        Args:
            query: 쿼리 벡터 (dim,)
            top_k: 반환할 결과 수

        Returns:
            List of (similarity, metadata) tuples
        """
        query = query.reshape(1, -1).astype(np.float32)

        if self.metric == 'cosine':
            query = self._normalize(query)

        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                results.append((float(dist), self.metadata[idx]))

        return results

    def save(self, path: str):
        """DB 저장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / 'index.faiss'))

        with open(path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        config = {
            'dim': self.dim,
            'index_type': self.index_type,
            'metric': self.metric,
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f)

    def load(self, path: str):
        """DB 로드"""
        path = Path(path)

        with open(path / 'config.json', 'r') as f:
            config = json.load(f)

        self.dim = config['dim']
        self.index_type = config['index_type']
        self.metric = config['metric']

        self.index = faiss.read_index(str(path / 'index.faiss'))

        with open(path / 'metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def __len__(self) -> int:
        return self.index.ntotal
