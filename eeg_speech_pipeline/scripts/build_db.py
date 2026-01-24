"""벡터 DB 구축 스크립트"""

import numpy as np
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import EEGSpeechPipeline


def main():
    # 1. 파이프라인 초기화
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    pipeline = EEGSpeechPipeline.from_config(str(config_path))

    # 2. 데이터 로드
    data_path = Path(__file__).parent.parent / 'data' / 'processed'

    eeg_data = np.load(data_path / 'eeg_epochs.npy')  # (n_samples, n_channels, n_timepoints)

    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # 3. 캘리브레이션 (첫 50개 샘플 사용)
    n_calib = min(50, len(eeg_data))
    calibration_samples = eeg_data[:n_calib]
    pipeline.calibrate(calibration_samples)

    # 4. DB 구축
    pipeline.build_database(eeg_data, metadata)

    # 5. 저장
    save_path = Path(__file__).parent.parent / 'data' / 'pipeline'
    pipeline.save(str(save_path))

    print("DB 구축 완료!")


if __name__ == '__main__':
    main()
