"""Tests for EEG Speech Decoding Pipeline."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessor import EEGPreprocessor
from src.encoders.wavelet_encoder import WaveletEncoder
from src.encoders.riemannian_encoder import RiemannianEncoder
from src.combiner import AutoWeightedCombiner


# Test constants
N_CHANNELS = 64
N_SAMPLES = 512
SAMPLING_RATE = 512


def make_eeg(n_channels=N_CHANNELS, n_samples=N_SAMPLES):
    """Generate synthetic EEG data for testing."""
    np.random.seed(42)
    return np.random.randn(n_channels, n_samples)


def make_eeg_batch(n_trials=10, n_channels=N_CHANNELS, n_samples=N_SAMPLES):
    """Generate batch of synthetic EEG data."""
    np.random.seed(42)
    return np.random.randn(n_trials, n_channels, n_samples)


class TestEEGPreprocessor:
    def test_init(self):
        prep = EEGPreprocessor(sampling_rate=SAMPLING_RATE)
        assert prep.sampling_rate == SAMPLING_RATE
        assert prep.lowcut == 0.5
        assert prep.highcut == 50.0
        assert prep.notch_freq == 60.0

    def test_process_shape(self):
        prep = EEGPreprocessor(sampling_rate=SAMPLING_RATE)
        eeg = make_eeg()
        result = prep.process(eeg)
        assert result.shape == eeg.shape

    def test_process_normalization(self):
        prep = EEGPreprocessor(sampling_rate=SAMPLING_RATE)
        eeg = make_eeg()
        result = prep.process(eeg)
        # After z-score, mean should be ~0 and std should be ~1
        assert np.abs(result.mean(axis=1)).max() < 0.1
        assert np.abs(result.std(axis=1) - 1.0).max() < 0.1

    def test_process_batch(self):
        prep = EEGPreprocessor(sampling_rate=SAMPLING_RATE)
        batch = make_eeg_batch(n_trials=5)
        result = prep.process_batch(batch)
        assert result.shape == batch.shape


class TestWaveletEncoder:
    def test_init(self):
        enc = WaveletEncoder(wavelet='db4', level=5)
        assert enc.wavelet == 'db4'
        assert enc.level == 5
        assert len(enc.features) == 5

    def test_encode_shape(self):
        enc = WaveletEncoder(wavelet='db4', level=5)
        eeg = make_eeg()
        result = enc.encode(eeg)
        expected_dim = enc.get_output_dim(N_CHANNELS)
        assert result.shape == (expected_dim,)

    def test_get_output_dim(self):
        enc = WaveletEncoder(wavelet='db4', level=5, features=['mean', 'std', 'rms', 'energy', 'entropy'])
        dim = enc.get_output_dim(N_CHANNELS)
        # 64 channels * (5+1) levels * 5 features = 1920
        assert dim == N_CHANNELS * 6 * 5

    def test_encode_deterministic(self):
        enc = WaveletEncoder(wavelet='db4', level=5)
        eeg = make_eeg()
        r1 = enc.encode(eeg)
        r2 = enc.encode(eeg)
        np.testing.assert_array_equal(r1, r2)

    def test_encode_batch(self):
        enc = WaveletEncoder(wavelet='db4', level=5)
        batch = make_eeg_batch(n_trials=3)
        result = enc.encode_batch(batch)
        expected_dim = enc.get_output_dim(N_CHANNELS)
        assert result.shape == (3, expected_dim)


class TestRiemannianEncoder:
    def test_init(self):
        enc = RiemannianEncoder(metric='riemann', n_iter=10)
        assert enc.metric == 'riemann'
        assert not enc._is_fitted

    def test_fit(self):
        # Use small channel count for speed
        n_ch = 8
        enc = RiemannianEncoder(metric='riemann', n_iter=5)
        batch = make_eeg_batch(n_trials=10, n_channels=n_ch, n_samples=256)
        enc.fit(batch)
        assert enc._is_fitted
        assert enc.reference_.shape == (n_ch, n_ch)

    def test_encode_without_fit_raises(self):
        enc = RiemannianEncoder()
        eeg = make_eeg(n_channels=8)
        with pytest.raises(RuntimeError):
            enc.encode(eeg)

    def test_encode_shape(self):
        n_ch = 8
        enc = RiemannianEncoder(metric='riemann', n_iter=5)
        batch = make_eeg_batch(n_trials=10, n_channels=n_ch, n_samples=256)
        enc.fit(batch)

        eeg = batch[0]
        result = enc.encode(eeg)
        expected_dim = enc.get_output_dim(n_ch)
        assert result.shape == (expected_dim,)

    def test_get_output_dim(self):
        enc = RiemannianEncoder()
        # n*(n+1)/2
        assert enc.get_output_dim(64) == 64 * 65 // 2  # 2080
        assert enc.get_output_dim(8) == 8 * 9 // 2  # 36

    def test_encode_real_valued(self):
        n_ch = 8
        enc = RiemannianEncoder(metric='riemann', n_iter=5)
        batch = make_eeg_batch(n_trials=10, n_channels=n_ch, n_samples=256)
        enc.fit(batch)

        result = enc.encode(batch[0])
        assert result.dtype in [np.float64, np.float32]
        assert not np.any(np.isnan(result))


class TestAutoWeightedCombiner:
    def test_init(self):
        comb = AutoWeightedCombiner(wav_dim=1920, riem_dim=2080, target_wav_ratio=0.5)
        assert comb.combined_dim == 1920 + 2080

    def test_scales(self):
        comb = AutoWeightedCombiner(wav_dim=100, riem_dim=200, target_wav_ratio=0.5)
        assert np.isclose(comb.w_scale, np.sqrt(0.5))
        assert np.isclose(comb.r_scale, np.sqrt(0.5))

    def test_combine_shape(self):
        wav_dim, riem_dim = 100, 200
        comb = AutoWeightedCombiner(wav_dim=wav_dim, riem_dim=riem_dim)
        wav = np.random.randn(wav_dim)
        riem = np.random.randn(riem_dim)
        result = comb.combine(wav, riem)
        assert result.shape == (wav_dim + riem_dim,)

    def test_combine_normalized_parts(self):
        wav_dim, riem_dim = 100, 200
        comb = AutoWeightedCombiner(wav_dim=wav_dim, riem_dim=riem_dim, target_wav_ratio=0.5)
        wav = np.random.randn(wav_dim)
        riem = np.random.randn(riem_dim)
        result = comb.combine(wav, riem)

        # Check that the wav part has norm = w_scale
        wav_part = result[:wav_dim]
        riem_part = result[wav_dim:]
        assert np.isclose(np.linalg.norm(wav_part), comb.w_scale, atol=1e-6)
        assert np.isclose(np.linalg.norm(riem_part), comb.r_scale, atol=1e-6)

    def test_set_ratio(self):
        comb = AutoWeightedCombiner(wav_dim=100, riem_dim=200, target_wav_ratio=0.5)
        comb.set_ratio(0.7)
        assert np.isclose(comb.w_scale, np.sqrt(0.7))
        assert np.isclose(comb.r_scale, np.sqrt(0.3))

    def test_combine_batch(self):
        wav_dim, riem_dim = 100, 200
        comb = AutoWeightedCombiner(wav_dim=wav_dim, riem_dim=riem_dim)
        wav_batch = np.random.randn(5, wav_dim)
        riem_batch = np.random.randn(5, riem_dim)
        result = comb.combine_batch(wav_batch, riem_batch)
        assert result.shape == (5, wav_dim + riem_dim)

    def test_analyze_similarity(self):
        wav_dim, riem_dim = 100, 200
        comb = AutoWeightedCombiner(wav_dim=wav_dim, riem_dim=riem_dim, target_wav_ratio=0.5)
        wav_a = np.random.randn(wav_dim)
        wav_b = wav_a + 0.1 * np.random.randn(wav_dim)  # Similar
        riem_a = np.random.randn(riem_dim)
        riem_b = np.random.randn(riem_dim)  # Different

        analysis = comb.analyze_similarity(wav_a, wav_b, riem_a, riem_b)
        assert 'wav_similarity' in analysis
        assert 'riem_similarity' in analysis
        assert 'combined_similarity' in analysis
        # wav should be more similar than riem
        assert analysis['wav_similarity'] > analysis['riem_similarity']

    def test_get_config(self):
        comb = AutoWeightedCombiner(wav_dim=100, riem_dim=200, target_wav_ratio=0.6)
        config = comb.get_config()
        assert config['wav_dim'] == 100
        assert config['riem_dim'] == 200
        assert config['combined_dim'] == 300
        assert np.isclose(config['target_wav_ratio'], 0.6)


class TestIntegration:
    """Integration tests with small channel count for speed."""

    def test_preprocess_to_wavelet(self):
        prep = EEGPreprocessor(sampling_rate=SAMPLING_RATE)
        enc = WaveletEncoder(wavelet='db4', level=5)

        eeg = make_eeg()
        processed = prep.process(eeg)
        embedding = enc.encode(processed)

        assert embedding.shape == (enc.get_output_dim(N_CHANNELS),)
        assert not np.any(np.isnan(embedding))

    def test_preprocess_to_riemannian(self):
        n_ch = 8
        prep = EEGPreprocessor(sampling_rate=SAMPLING_RATE)
        enc = RiemannianEncoder(metric='riemann', n_iter=5)

        batch = make_eeg_batch(n_trials=10, n_channels=n_ch, n_samples=256)
        processed = prep.process_batch(batch)
        enc.fit(processed)

        result = enc.encode(processed[0])
        assert result.shape == (enc.get_output_dim(n_ch),)

    def test_full_encoding_pipeline(self):
        n_ch = 8
        prep = EEGPreprocessor(sampling_rate=SAMPLING_RATE)
        wav_enc = WaveletEncoder(wavelet='db4', level=5)
        riem_enc = RiemannianEncoder(metric='riemann', n_iter=5)

        wav_dim = wav_enc.get_output_dim(n_ch)
        riem_dim = riem_enc.get_output_dim(n_ch)
        combiner = AutoWeightedCombiner(wav_dim=wav_dim, riem_dim=riem_dim)

        # Calibrate
        batch = make_eeg_batch(n_trials=10, n_channels=n_ch, n_samples=256)
        processed = prep.process_batch(batch)
        riem_enc.fit(processed)

        # Encode
        eeg = processed[0]
        wav_emb = wav_enc.encode(eeg)
        riem_emb = riem_enc.encode(eeg)
        combined = combiner.combine(wav_emb, riem_emb)

        assert combined.shape == (wav_dim + riem_dim,)
        assert not np.any(np.isnan(combined))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
