"""Wavelet-based EEG feature encoder."""

import numpy as np
import pywt
from typing import List


class WaveletEncoder:
    """
    Wavelet 기반 EEG 특징 추출기

    학습 불필요 - 순수 신호 처리 기반

    처리:
    1. 각 채널에 Wavelet 분해 (db4, 5레벨)
    2. 각 레벨에서 통계 특징 추출
    3. 전체 채널 concatenation

    Args:
        wavelet: Wavelet 종류 (default: 'db4')
        level: 분해 레벨 (default: 5)
        features: 추출할 통계 특징 리스트
    """

    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 5,
        features: List[str] = None,
    ):
        self.wavelet = wavelet
        self.level = level
        self.features = features or ['mean', 'std', 'rms', 'energy', 'entropy']

    def _compute_features(self, coeffs: np.ndarray) -> np.ndarray:
        """단일 계수 배열에서 통계 특징 추출"""
        feats = []

        if 'mean' in self.features:
            feats.append(np.mean(coeffs))
        if 'std' in self.features:
            feats.append(np.std(coeffs))
        if 'rms' in self.features:
            feats.append(np.sqrt(np.mean(coeffs**2)))
        if 'energy' in self.features:
            feats.append(np.sum(coeffs**2))
        if 'entropy' in self.features:
            p = np.abs(coeffs) / (np.sum(np.abs(coeffs)) + 1e-10)
            feats.append(-np.sum(p * np.log(p + 1e-10)))

        return np.array(feats)

    def encode_channel(self, channel_data: np.ndarray) -> np.ndarray:
        """단일 채널 인코딩"""
        coeffs = pywt.wavedec(channel_data, self.wavelet, level=self.level)

        all_features = []
        for coeff in coeffs:  # cA5, cD5, cD4, cD3, cD2, cD1
            all_features.extend(self._compute_features(coeff))

        return np.array(all_features)

    def encode(self, eeg: np.ndarray) -> np.ndarray:
        """
        EEG 인코딩

        Args:
            eeg: 전처리된 EEG (n_channels, n_samples)

        Returns:
            embedding: Wavelet 임베딩 벡터
        """
        n_channels = eeg.shape[0]

        channel_features = []
        for ch in range(n_channels):
            ch_feat = self.encode_channel(eeg[ch])
            channel_features.append(ch_feat)

        # 전체 concat → 큰 차원
        embedding = np.concatenate(channel_features)

        return embedding

    def encode_batch(self, eeg_batch: np.ndarray) -> np.ndarray:
        """배치 인코딩"""
        return np.array([self.encode(eeg) for eeg in eeg_batch])

    def get_output_dim(self, n_channels: int) -> int:
        """출력 차원 계산"""
        features_per_level = len(self.features)
        levels = self.level + 1  # approximation + details
        return n_channels * levels * features_per_level
