"""추론 스크립트"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import EEGSpeechPipeline


def main():
    # 1. 파이프라인 로드
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    pipeline = EEGSpeechPipeline.from_config(str(config_path))

    pipeline_path = Path(__file__).parent.parent / 'data' / 'pipeline'
    pipeline.load(str(pipeline_path))

    # 2. 새 EEG 샘플 (예시 - 실제로는 실시간 EEG 입력)
    n_channels = pipeline.config['data']['n_channels']
    sampling_rate = pipeline.config['data']['sampling_rate']
    new_eeg = np.random.randn(n_channels, sampling_rate)

    # 3. 예측
    result = pipeline.predict_with_details(new_eeg, top_k=5)

    print(f"생성된 문장: {result['text']}")
    print(f"최상위 의도: {result['top_intent']}")
    print(f"유사도: {result['top_similarity']:.3f}")

    print("\n검색 결과:")
    for score, meta in result['search_results']:
        print(f"  [{score:.3f}] {meta['intent']}: {meta['candidate_texts'][0]}")


if __name__ == '__main__':
    main()
