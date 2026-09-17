import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from app.db.session import async_session_maker
from app.graph import build_graph
from app.storage import save_message

app = FastAPI(title="Travel Planning Agent")


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    reply: str
    itinerary: dict | None = None
    new_session_id: str | None = None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or session_id  # fallback: treat as their own user

    async with async_session_maker() as db:
        await save_message(db, session_id, "user", request.message)

        graph = build_graph(db)
        state = {
            "session_id": session_id,
            "user_id": user_id,
            "last_user_message": request.message,
        }
        result = await graph.ainvoke(state)

        itinerary = result.get("itinerary")
        reply = "" if itinerary else result.get("assistant_reply", "")

        content_to_save = reply if reply else str(itinerary)
        await save_message(db, session_id, "assistant", content_to_save)

    return ChatResponse(
        session_id=session_id,
        user_id=user_id,
        reply=reply,
        itinerary=itinerary,
        new_session_id=result.get("new_session_id"),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}