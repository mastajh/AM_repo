"""Riemannian geometry-based EEG feature encoder."""

import numpy as np
from scipy import linalg


class RiemannianEncoder:
    """
    Riemannian geometry 기반 EEG 특징 추출기

    fit() 필요 - 참조 행렬(리만 평균) 계산에 캘리브레이션 데이터 필요

    처리:
    1. 공분산 행렬 계산
    2. 탄젠트 공간 투영 (참조 행렬 기준)
    3. 상삼각 요소 추출

    Args:
        metric: 거리 메트릭 ('riemann', 'logeuclid', 'euclid')
        n_iter: 리만 평균 계산 반복 횟수
    """

    def __init__(
        self,
        metric: str = 'riemann',
        n_iter: int = 50,
    ):
        self.metric = metric
        self.n_iter = n_iter
        self.reference_ = None
        self._is_fitted = False

    def _compute_covariance(self, eeg: np.ndarray) -> np.ndarray:
        """공분산 행렬 계산 (정규화 포함)"""
        n_samples = eeg.shape[1]
        cov = eeg @ eeg.T / n_samples

        # 정규화 (trace = n_channels)
        cov = cov / np.trace(cov) * eeg.shape[0]

        # 양정치 보장
        cov = cov + 1e-6 * np.eye(cov.shape[0])

        return cov

    def _riemann_mean(self, covs: np.ndarray) -> np.ndarray:
        """
        리만 평균 계산 (iterative algorithm)

        Args:
            covs: 공분산 행렬들 (n_samples, n_channels, n_channels)
        """
        n_samples, n_ch, _ = covs.shape

        # 초기값: 산술 평균
        mean = np.mean(covs, axis=0)

        for _ in range(self.n_iter):
            mean_sqrt_inv = linalg.inv(linalg.sqrtm(mean))

            tangent_sum = np.zeros_like(mean)
            for cov in covs:
                M = mean_sqrt_inv @ cov @ mean_sqrt_inv
                tangent_sum += linalg.logm(M)

            tangent_mean = tangent_sum / n_samples

            mean_sqrt = linalg.sqrtm(mean)
            mean = mean_sqrt @ linalg.expm(tangent_mean) @ mean_sqrt

        return mean

    def fit(self, eeg_samples: np.ndarray) -> 'RiemannianEncoder':
        """
        참조 행렬 계산 (캘리브레이션)

        Args:
            eeg_samples: 캘리브레이션 EEG 샘플들 (n_samples, n_channels, n_timepoints)

        Returns:
            self
        """
        covs = np.array([self._compute_covariance(eeg) for eeg in eeg_samples])

        self.reference_ = self._riemann_mean(covs)
        self._is_fitted = True

        return self

    def _tangent_space(self, cov: np.ndarray) -> np.ndarray:
        """탄젠트 공간 투영"""
        if not self._is_fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        ref_sqrt_inv = linalg.inv(linalg.sqrtm(self.reference_))
        M = ref_sqrt_inv @ cov @ ref_sqrt_inv

        tangent = linalg.logm(M)

        return tangent

    def _upper_triangular(self, matrix: np.ndarray) -> np.ndarray:
        """상삼각 요소 추출 (대각 포함)"""
        n = matrix.shape[0]
        indices = np.triu_indices(n)

        result = matrix[indices].copy()
        # 비대각 요소는 √2 스케일링 (frobenius norm 보존)
        diag_mask = indices[0] != indices[1]
        result[diag_mask] *= np.sqrt(2)

        return result

    def encode(self, eeg: np.ndarray) -> np.ndarray:
        """
        EEG 인코딩

        Args:
            eeg: 전처리된 EEG (n_channels, n_samples)

        Returns:
            embedding: Riemannian 임베딩 벡터
        """
        cov = self._compute_covariance(eeg)
        tangent = self._tangent_space(cov)
        embedding = self._upper_triangular(tangent)

        return embedding.real  # 허수부 제거

    def encode_batch(self, eeg_batch: np.ndarray) -> np.ndarray:
        """배치 인코딩"""
        return np.array([self.encode(eeg) for eeg in eeg_batch])

    def get_output_dim(self, n_channels: int) -> int:
        """출력 차원 계산: n*(n+1)/2"""
        return n_channels * (n_channels + 1) // 2

    def save(self, path: str):
        """참조 행렬 저장"""
        np.save(path, self.reference_)

    def load(self, path: str):
        """참조 행렬 로드"""
        self.reference_ = np.load(path)
        self._is_fitted = True
