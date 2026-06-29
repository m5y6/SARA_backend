"""
LLM service using Google GenAI SDK.
Generates responses based on retrieved document fragments and user questions.
"""

from google import genai
from typing import List, Dict, Any, Optional
from app.core.config import settings
from google.genai import types

class GeminiService:
    """Service for interacting with Google GenAI SDK."""

    def __init__(self):
        try:
            self.client = genai.Client(api_key=settings.gemini_api_key)
        except Exception as e:
            raise ValueError("Failed to initialize Gemini client. Is GEMINI_API_KEY set?") from e
        self.model_name = f"models/{settings.llm_model}"

    def _build_system_prompt(self, has_fragments: bool = True, has_history: bool = False) -> str:
        if has_fragments:
            base_prompt = """You are SARA (Sistema de Asistencia de Reglamentos Académicos), an AI support assistant for DUOC UC.

You have been provided with one or more fragments from DUOC UC's academic regulations to help you answer the user's question.

Your task is to generate a helpful and accurate response by following these instructions:

1.  **Critically Analyze the Fragments**: First, carefully review the provided fragments. Determine if they actually contain information that is relevant to the user's specific question.
2.  **Synthesize an Answer (if relevant)**: If the fragments contain relevant information, synthesize a concise answer based **only** on what is written in the fragments. When you use information from a fragment, cite it by its title (e.g., "[Título del Documento]").
3.  **Handle Irrelevant Fragments**: If you determine that the provided fragments are NOT relevant to the question, you MUST NOT use them. Instead, you should state that you couldn't find a specific answer in the provided documents and recommend the user consult official DUOC UC sources. Do not invent an answer or use external knowledge.
4.  **Language**: Always respond in Spanish.
"""
        else:
            base_prompt = """You are SARA (Sistema de Asistencia de Reglamentos Académicos), an AI support assistant for DUOC UC.

INSTRUCTIONS:
1. No relevant regulation fragments were found in the database
2. You can answer general questions about DUOC UC policies, but be cautious
3. Suggest the user consult official DUOC UC documentation if unsure
4. Always respond as an official DUOC UC support assistant
5. Respond in Spanish"""

        if has_history:
            base_prompt += """
6. **Contextual Awareness**: You have been provided with the recent history of the conversation. Use this history to understand the context of the user's current question, especially if it is short or ambiguous (e.g., 'and for grades?'). Avoid repeating greetings if a conversation is already in progress."""
        return base_prompt

    def _build_user_prompt(self, question: str, fragments: List[Dict[str, Any]], chat_history: Optional[str] = None) -> str:
        has_fragments = bool(fragments)
        has_history = bool(chat_history)
        prompt = self._build_system_prompt(has_fragments=has_fragments, has_history=has_history)

        if chat_history:
            prompt += f"""

CONVERSATION HISTORY:
{chat_history}"""

        if fragments:
            prompt += """

RELEVANT DOCUMENTS:
"""
            for i, fragment in enumerate(fragments, 1):
                similarity = fragment.get("similarity", 0)
                titulo = fragment.get("titulo") or f"Document {i}"
                contenido = fragment.get("contenido_texto") or fragment.get("contenido", "")
                prompt += f"""
[{titulo}] (Relevance: {similarity:.1%})
{contenido}
"""
        
        prompt += f"""
---
CURRENT QUESTION: {question}"""
        return prompt
    
    def _call_gemini_sdk(self, prompt: str, temperature: float, max_tokens: int) -> str:
        config_obj = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config_obj
            )
            try:
                return response.text
            except ValueError:
                # If the response doesn't contain text, check if the prompt was blocked.
                if response.prompt_feedback.block_reason:
                     raise ValueError(f"Request was blocked by the API: {response.prompt_feedback.block_reason.name}")
                else:
                    raise ValueError("Empty response from Gemini API and no block reason.")

        except Exception as e:
            raise Exception(f"Google GenAI SDK call failed: {str(e)}") from e

    def generate_response(
        self,
        question: str,
        fragments: List[Dict[str, Any]],
        chat_history: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        max_tokens = max_tokens or settings.max_tokens
        user_prompt = self._build_user_prompt(question, fragments, chat_history)
        
        return self._call_gemini_sdk(user_prompt, temperature, max_tokens)

    def generate_rag_response(
        self,
        question: str,
        fragments: List[Dict[str, Any]],
        chat_history: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Generate response with or without fragments - let Gemini handle both cases."""
        answer = self.generate_response(
            question=question,
            fragments=fragments,
            chat_history=chat_history,
            temperature=temperature
        )
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
        rag_response = self.generate_rag_response(
            question=question,
            fragments=[],
            chat_history=None, # No history in fallback
            temperature=temperature
        )
        return rag_response


_gemini_service: Optional[GeminiService] = None

def get_gemini_service() -> GeminiService:
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service
