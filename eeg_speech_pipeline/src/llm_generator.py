"""RAG-based sentence generator using LLM."""

import os
from typing import List, Dict, Any, Optional, Tuple


class LLMGenerator:
    """
    RAG 기반 문장 생성기

    검색된 유사 의도들을 바탕으로 LLM이 최종 문장 생성

    Args:
        model: 모델 이름
        api_key: API 키 (환경변수에서 로드 가능)
        temperature: 생성 온도
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            self.client = None

    def _build_prompt(
        self,
        search_results: List[Tuple[float, Dict[str, Any]]],
        context: Optional[str] = None,
    ) -> str:
        """RAG 프롬프트 구성"""
        candidates = []
        for i, (score, meta) in enumerate(search_results, 1):
            intent = meta.get('intent', 'unknown')
            texts = meta.get('candidate_texts', [])
            candidates.append(f"{i}. 의도: {intent} (유사도: {score:.3f})")
            for text in texts[:3]:
                candidates.append(f"   - {text}")

        candidates_str = "\n".join(candidates)

        prompt = f"""당신은 EEG 신호에서 추출한 의도를 자연스러운 한국어 문장으로 변환하는 도우미입니다.

검색된 유사 의도들:
{candidates_str}

위 의도들을 참고하여, 사용자가 말하고 싶어하는 가장 적절한 문장 하나를 생성하세요.
- 간결하고 자연스러운 한국어로 작성
- 가장 유사도가 높은 의도를 우선 고려
- 필요시 여러 의도를 조합

생성할 문장:"""

        if context:
            prompt = f"상황: {context}\n\n" + prompt

        return prompt

    def generate(
        self,
        search_results: List[Tuple[float, Dict[str, Any]]],
        context: Optional[str] = None,
    ) -> str:
        """
        문장 생성

        Args:
            search_results: 벡터 DB 검색 결과 [(score, metadata), ...]
            context: 추가 상황 정보

        Returns:
            generated_text: 생성된 문장
        """
        prompt = self._build_prompt(search_results, context)

        if self.client is None:
            # 더미 모드: 가장 유사한 결과의 첫 번째 텍스트 반환
            if search_results:
                texts = search_results[0][1].get('candidate_texts', [])
                return texts[0] if texts else "알 수 없음"
            return "알 수 없음"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 EEG 기반 의사소통 보조 시스템입니다."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=100,
        )

        return response.choices[0].message.content.strip()

    def generate_batch(
        self,
        search_results_batch: List[List[Tuple[float, Dict[str, Any]]]],
        contexts: Optional[List[str]] = None,
    ) -> List[str]:
        """배치 생성"""
        if contexts is None:
            contexts = [None] * len(search_results_batch)

        return [
            self.generate(results, ctx)
            for results, ctx in zip(search_results_batch, contexts)
        ]
