from app.llm import _get_llm

SYSTEM_PROMPT = """You are the retrieval-response step of a travel planning agent.
The user is asking about their already-generated trip itinerary.
Answer their question using ONLY the itinerary data provided below.
If the itinerary doesn't have the information they're asking about, say so
honestly. Keep the answer short and conversational. Respond with plain text
only, no JSON, no markdown formatting.
"""


async def run_retrieve(user_message: str, itinerary: dict) -> str:
    if not itinerary:
        return "You don't have a generated itinerary yet. Want to plan a new trip?"

    user_content = f"""
Saved itinerary: {itinerary}

User's question: {user_message}
"""
    llm = _get_llm()
    response = await llm.ainvoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )
    return response.content