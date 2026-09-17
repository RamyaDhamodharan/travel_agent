from typing import Type, TypeVar

from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.config import settings

T = TypeVar("T", bound=BaseModel)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )
    return _llm


async def call_structured(system_prompt: str, user_content: str, schema: Type[T]) -> T:
    structured_llm = _get_llm().with_structured_output(schema)
    result = await structured_llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    )
    return result