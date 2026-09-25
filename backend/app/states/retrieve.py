from app.llm import _get_llm

SYSTEM_PROMPT = """You are the retrieval-response step of a travel planning agent.
The user is asking about a trip's already-generated itinerary.
Answer their question using ONLY the itinerary data provided below.
If the itinerary doesn't have the information they're asking about, say so
honestly. Keep the answer short and conversational. Respond with plain text
only, no JSON, no markdown formatting.

FEASIBILITY QUESTIONS (e.g. "is it possible in 500", "can this be done in
1000"): judge against the itinerary's estimated_total_cost, budget_breakdown,
and notes. If the asked amount is NOT feasible, say so clearly and always
give a concrete minimum realistic amount they'd need instead (pull it from
notes/estimated_total_cost if stated, or your best estimate otherwise) --
never just say "not possible" without naming a number to aim for.
"""


async def run_retrieve(user_message: str, itinerary: dict | None) -> str:
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