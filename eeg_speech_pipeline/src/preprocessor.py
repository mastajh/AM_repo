"""EEG signal preprocessor module."""

import numpy as np
from scipy import signal


class EEGPreprocessor:
    """
    EEG 신호 전처리기

    처리 순서:
    1. Bandpass filter (0.5-50Hz)
    2. Notch filter (60Hz)
    3. Z-score normalization (채널별)

    Args:
        sampling_rate: 샘플링 레이트 (Hz)
        lowcut: 밴드패스 하한 (Hz)
        highcut: 밴드패스 상한 (Hz)
        notch_freq: 노치 필터 주파수 (Hz)
    """

    def __init__(
        self,
        sampling_rate: int = 512,
        lowcut: float = 0.5,
        highcut: float = 50.0,
        notch_freq: float = 60.0,
    ):
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq

        self._init_filters()

    def _init_filters(self):
        """필터 계수 초기화"""
        nyq = self.sampling_rate / 2

        # Bandpass filter (4th order Butterworth)
        self.b_bp, self.a_bp = signal.butter(
            4, [self.lowcut / nyq, self.highcut / nyq], btype='band'
        )

        # Notch filter (Q=30)
        self.b_notch, self.a_notch = signal.iirnotch(
            self.notch_freq / nyq, Q=30
        )

    def process(self, eeg: np.ndarray) -> np.ndarray:
        """
        EEG 전처리 수행

        Args:
            eeg: 원본 EEG 데이터 (n_channels, n_samples)

        Returns:
            processed: 전처리된 EEG (n_channels, n_samples)
        """
        # 1. Bandpass filter
        filtered = signal.filtfilt(self.b_bp, self.a_bp, eeg, axis=1)

        # 2. Notch filter
        filtered = signal.filtfilt(self.b_notch, self.a_notch, filtered, axis=1)

        # 3. Z-score normalization (채널별)
        mean = filtered.mean(axis=1, keepdims=True)
        std = filtered.std(axis=1, keepdims=True) + 1e-10
        normalized = (filtered - mean) / std

        return normalized

    def process_batch(self, eeg_batch: np.ndarray) -> np.ndarray:
        """
        배치 전처리

        Args:
            eeg_batch: (n_samples, n_channels, n_timepoints)

        Returns:
            processed: (n_samples, n_channels, n_timepoints)
        """
        return np.array([self.process(eeg) for eeg in eeg_batch])
