"""
LLM service using Google Gemini API.
Generates responses based on retrieved document fragments and user questions.
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional
from app.core.config import settings


class GeminiService:
    """Service for interacting with Google Gemini API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = settings.llm_model
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _get_fallback_model_names(self) -> List[str]:
        candidates = [
            self.model_name,
            "gemini-2.5-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash",
            "gemini-flash-latest",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro",
        ]
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
        return unique_candidates

    def _build_system_prompt(self) -> str:
        return """You are SARA (Sistema de Asistencia de Reglamentos Académicos), 
an expert assistant for academic regulations.

CONSTRAINTS:
1. Answer ONLY based on provided fragments.
2. If not found, state: "No encontré información relevante sobre esta pregunta en los reglamentos académicos disponibles."
3. Be concise and cite relevant sections.
4. Use Spanish language.

Document Fragments:
"""

    def _build_user_prompt(self, question: str, fragments: List[Dict[str, Any]]) -> str:
        prompt = self._build_system_prompt()
        if fragments:
            for i, fragment in enumerate(fragments, 1):
                similarity = fragment.get("similarity", 0)
                contenido = fragment.get("contenido", "")
                prompt += f"\n[Fragment {i} - Relevance: {similarity:.2%}]\n{contenido}\n"
        else:
            prompt += "\n[No relevant fragments found]\n"
        prompt += f"\n---\n\nUser Question: {question}\n\nAnswer:"
        return prompt

    def generate_response(
        self,
        question: str,
        fragments: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        max_tokens = max_tokens or settings.max_tokens
        user_prompt = self._build_user_prompt(question, fragments)
        last_error: Optional[Exception] = None

        for model_name in self._get_fallback_model_names():
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    user_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                if not response.text:
                    raise ValueError("Empty response from Gemini API")
                self.model_name = model_name
                self.model = model
                return response.text
            except Exception as e:
                last_error = e
                message = str(e).lower()
                retryable = (
                    "404" in message
                    or "not found" in message
                    or "not supported" in message
                    or "quota" in message
                    or "rate limit" in message
                    or "exceeded" in message
                )
                if not retryable:
                    break

        raise Exception(f"Gemini API call failed: {str(last_error)}")

    def generate_rag_response(
        self,
        question: str,
        fragments: List[Dict[str, Any]],
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        answer = self.generate_response(question=question, fragments=fragments, temperature=temperature)
        return {
            "answer": answer,
            "fragments_used": len(fragments),
            "fragments": fragments,
        }


_gemini_service: GeminiService = None

def get_gemini_service() -> GeminiService:
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service
