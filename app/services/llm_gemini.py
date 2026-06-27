"""
LLM service using Google Gemini API.
Generates responses based on retrieved document fragments and user questions.
"""

import google.generativeai as genai
import re
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

    def _build_system_prompt(self, has_fragments: bool = True) -> str:
        if has_fragments:
            return """You are SARA (Sistema de Asistencia de Reglamentos Académicos), an AI support assistant for DUOC UC.

You have been provided with one or more fragments from DUOC UC's academic regulations to help you answer the user's question.

Your task is to generate a helpful and accurate response by following these instructions:

1.  **Critically Analyze the Fragments**: First, carefully review the provided fragments. Determine if they actually contain information that is relevant to the user's specific question.
2.  **Synthesize an Answer (if relevant)**: If the fragments contain relevant information, synthesize a concise answer based **only** on what is written in the fragments. When you use information from a fragment, cite it by its title (e.g., "[Título del Documento]").
3.  **Handle Irrelevant Fragments**: If you determine that the provided fragments are NOT relevant to the question, you MUST NOT use them. Instead, you should state that you couldn't find a specific answer in the provided documents and recommend the user consult official DUOC UC sources. Do not invent an answer or use external knowledge.
4.  **Language**: Always respond in Spanish.
"""
        else:
            return """You are SARA (Sistema de Asistencia de Reglamentos Académicos), an AI support assistant for DUOC UC.

INSTRUCTIONS:
1. No relevant regulation fragments were found in the database
2. You can answer general questions about DUOC UC policies, but be cautious
3. Suggest the user consult official DUOC UC documentation if unsure
4. Always respond as an official DUOC UC support assistant
5. Respond in Spanish"""

    def _build_user_prompt(self, question: str, fragments: List[Dict[str, Any]]) -> str:
        has_fragments = bool(fragments)
        prompt = self._build_system_prompt(has_fragments=has_fragments)
        
        if fragments:
            prompt += "\n\nRELEVANT DOCUMENTS:\n"
            for i, fragment in enumerate(fragments, 1):
                similarity = fragment.get("similarity", 0)
                titulo = fragment.get("titulo") or f"Document {i}"
                contenido = fragment.get("contenido_texto") or fragment.get("contenido", "")
                prompt += f"\n[{titulo}] (Relevance: {similarity:.1%})\n{contenido}\n"
        
        prompt += f"\n---\nQUESTION: {question}"
        return prompt

    def _build_fallback_prompt(self, question: str) -> str:
        return f"""You are SARA (Sistema de Asistencia de Reglamentos Académicos), an AI support assistant for DUOC UC.

    No relevant fragments were found in the vectorized academic regulations database for this question.

    Instructions:
    1. Answer in Spanish.
    2. Respond naturally and helpfully, like a support assistant for DUOC UC.
    3. Do not mention internal retrieval errors or technical limitations.
    4. If the question is about academic or institutional rules, give the best general guidance you can without inventing specific regulations.
    5. If you are unsure about an exact policy, say it should be verified in the official DUOC UC documents.
    6. Keep the response concise and friendly.

    User Question: {question}

    Answer:"""

    @staticmethod
    def _is_refusal_answer(answer: str) -> bool:
        normalized = (answer or "").lower()
        refusal_markers = (
            "no encontré",
            "no encontre",
            "no contiene información relevante",
            "no contiene informacion relevante",
            "lo siento",
            "te sugiero revisar el documento oficial",
            "no relevant",
        )
        return any(marker in normalized for marker in refusal_markers)

    @staticmethod
    def _normalize_question_keywords(question: str) -> List[str]:
        normalized = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", (question or "").lower())
        stopwords = {
            "que",
            "tipo",
            "cuál",
            "cual",
            "puedo",
            "debo",
            "usar",
            "usaré",
            "puedo",
            "sobre",
            "para",
            "una",
            "un",
            "el",
            "la",
            "los",
            "las",
            "en",
            "y",
            "o",
            "me",
            "mi",
            "tu",
            "se",
            "ocupar",
            "utilizar",
            "usar",
            "presentacion",
            "presentación",
        }
        return [token for token in normalized.split() if len(token) > 3 and token not in stopwords]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", (text or "").lower())

    def _synthesize_fragment_answer(self, question: str, fragments: List[Dict[str, Any]]) -> str:
        question_keywords = set(self._normalize_question_keywords(question))
        
        best_sentences = []
        
        for fragment in fragments:
            titulo = fragment.get("titulo") or "fragmento"
            contenido = fragment.get("contenido_texto") or fragment.get("contenido", "")
            contenido = contenido.strip() if contenido else ""
            if not contenido:
                continue
            
            lines = contenido.split("\n")
            for line in lines:
                clean = line.replace("●", "").strip()
                if len(clean) < 30:
                    continue
                
                line_norm = self._normalize_text(clean)
                keyword_matches = sum(1 for kw in question_keywords if kw in line_norm)
                
                if keyword_matches > 0:
                    best_sentences.append((keyword_matches, len(clean), titulo, clean))
        
        if best_sentences:
            best_sentences.sort(key=lambda x: (-x[0], -x[1]))
            selected = best_sentences[:2]
            lines = ["De acuerdo a los reglamentos académicos:"]
            for _, _, titulo, sentence in selected:
                lines.append(f"• {sentence[:250]}")
            return "\n".join(lines)
        
        return "No encontré información suficiente en los fragmentos recuperados para responder con precisión."

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
        """Generate response with or without fragments - let Gemini handle both cases."""
        answer = self.generate_response(question=question, fragments=fragments, temperature=temperature)
        return {
            "answer": answer,
            "fragments_used": len(fragments),
            "fragments": fragments,
        }

    def generate_fallback_response(
        self,
        question: str,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        max_tokens = settings.max_tokens
        fallback_prompt = self._build_fallback_prompt(question)
        last_error: Optional[Exception] = None

        for model_name in self._get_fallback_model_names():
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    fallback_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                if not response.text:
                    raise ValueError("Empty response from Gemini API")
                self.model_name = model_name
                self.model = model
                return {
                    "answer": response.text,
                    "fragments_used": 0,
                    "fragments": [],
                }
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


_gemini_service: GeminiService = None

def get_gemini_service() -> GeminiService:
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service
