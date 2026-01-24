"""Auto-weighted embedding combiner for wavelet and Riemannian features."""

import numpy as np
from typing import Dict, Any


class AutoWeightedCombiner:
    """
    차원 수를 자동 보정하고 발언권 비율을 조절하는 임베딩 결합기

    핵심 공식:
        w_scale = √r
        r_scale = √(1-r)
        combined = [w_scale × L2_norm(wav), r_scale × L2_norm(riem)]

    코사인 유사도 분해:
        cos_sim = r × wav_sim + (1-r) × riem_sim

    Args:
        wav_dim: Wavelet 임베딩 차원
        riem_dim: Riemannian 임베딩 차원
        target_wav_ratio: Wavelet 발언권 비율 (0~1, 기본값 0.5)
    """

    def __init__(
        self,
        wav_dim: int,
        riem_dim: int,
        target_wav_ratio: float = 0.5,
    ):
        self.wav_dim = wav_dim
        self.riem_dim = riem_dim
        self.combined_dim = wav_dim + riem_dim

        self.set_ratio(target_wav_ratio)

    def set_ratio(self, target_wav_ratio: float) -> None:
        """발언권 비율 설정/변경"""
        self.target_wav_ratio = np.clip(target_wav_ratio, 0.01, 0.99)
        r = self.target_wav_ratio

        self.w_scale = np.sqrt(r)
        self.r_scale = np.sqrt(1 - r)

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 정규화"""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec
        return vec / norm

    def combine(
        self,
        wav_emb: np.ndarray,
        riem_emb: np.ndarray,
    ) -> np.ndarray:
        """
        두 임베딩을 가중 결합

        Args:
            wav_emb: Wavelet 임베딩 (wav_dim,)
            riem_emb: Riemannian 임베딩 (riem_dim,)

        Returns:
            combined: 결합된 임베딩 (combined_dim,)
        """
        wav_norm = self._l2_normalize(wav_emb)
        riem_norm = self._l2_normalize(riem_emb)

        combined = np.concatenate([
            self.w_scale * wav_norm,
            self.r_scale * riem_norm,
        ])

        return combined

    def combine_batch(
        self,
        wav_embs: np.ndarray,
        riem_embs: np.ndarray,
    ) -> np.ndarray:
        """배치 결합"""
        wav_norms = wav_embs / (np.linalg.norm(wav_embs, axis=1, keepdims=True) + 1e-10)
        riem_norms = riem_embs / (np.linalg.norm(riem_embs, axis=1, keepdims=True) + 1e-10)

        combined = np.concatenate([
            self.w_scale * wav_norms,
            self.r_scale * riem_norms,
        ], axis=1)

        return combined

    def analyze_similarity(
        self,
        wav_a: np.ndarray, wav_b: np.ndarray,
        riem_a: np.ndarray, riem_b: np.ndarray,
    ) -> Dict[str, float]:
        """두 샘플 간 유사도 상세 분석"""
        wav_sim = np.dot(self._l2_normalize(wav_a), self._l2_normalize(wav_b))
        riem_sim = np.dot(self._l2_normalize(riem_a), self._l2_normalize(riem_b))

        combined_a = self.combine(wav_a, riem_a)
        combined_b = self.combine(wav_b, riem_b)
        combined_sim = np.dot(
            self._l2_normalize(combined_a),
            self._l2_normalize(combined_b),
        )

        r = self.target_wav_ratio
        wav_contrib = r * wav_sim
        riem_contrib = (1 - r) * riem_sim
        total = wav_contrib + riem_contrib

        return {
            'wav_similarity': float(wav_sim),
            'riem_similarity': float(riem_sim),
            'combined_similarity': float(combined_sim),
            'wav_contribution': float(wav_contrib),
            'riem_contribution': float(riem_contrib),
            'wav_contribution_ratio': float(wav_contrib / total * 100) if total > 0 else 0,
            'riem_contribution_ratio': float(riem_contrib / total * 100) if total > 0 else 0,
        }

    def get_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        return {
            'wav_dim': self.wav_dim,
            'riem_dim': self.riem_dim,
            'combined_dim': self.combined_dim,
            'target_wav_ratio': self.target_wav_ratio,
            'w_scale': float(self.w_scale),
            'r_scale': float(self.r_scale),
        }
