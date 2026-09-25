import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel

from app.db.session import async_session_maker
from app.graph import build_graph
from app.storage import save_message, list_trips_for_user, load_trip, get_itinerary, get_history
from app.tools.places_search import get_places_guide

app = FastAPI(title="Travel Planning Agent")

# Allow the frontend (running on a different port/origin) to call this API.
# For local dev this is permissive; tighten allow_origins to your actual
# frontend URL(s) before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_AFFIRMATIVE = {"yes", "yeah", "yep", "ok", "okay", "confirm", "confirmed", "proceed", "sure", "go ahead", "fine", "correct", "looks good", "good"}


def _is_affirmative(message: str) -> bool:
    m = message.strip().lower()
    return any(m == w or m.startswith(w + " ") or m.startswith(w + ",") for w in _AFFIRMATIVE)


def _pending_interrupt_type(state_snapshot) -> str | None:
    """
    Reads the 'type' field out of the currently-paused interrupt's own
    payload (the dict we passed to interrupt() in the node), rather than
    tracking a separate state flag -- LangGraph already carries this info
    on the state snapshot's tasks, so we don't need extra graph nodes just
    to record it ourselves.
    """
    if not state_snapshot or not state_snapshot.tasks:
        return None
    for task in state_snapshot.tasks:
        if task.interrupts:
            payload = task.interrupts[0].value
            if isinstance(payload, dict):
                return payload.get("type")
    return None


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    reply: str
    itinerary: dict | None = None
    new_session_id: str | None = None
    interrupt: dict | None = None


def _extract_response(session_id: str, user_id: str, result: dict) -> ChatResponse:
    interrupts = result.get("__interrupt__")
    if interrupts:
        payload = interrupts[0].value
        return ChatResponse(session_id=session_id, user_id=user_id, reply="", interrupt=payload)

    itinerary = result.get("itinerary")
    reply = result.get("assistant_reply", "")
    return ChatResponse(
        session_id=session_id,
        user_id=user_id,
        reply=reply,
        itinerary=itinerary,
        new_session_id=result.get("new_session_id"),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Single entry point for everything: starting a new trip, sending a
    follow-up, or answering a paused confirmation. We check whether this
    session_id currently has a paused graph; if so, your message is
    interpreted as the answer to that pause (yes/confirm vs. a correction),
    otherwise it's treated as a normal new message.
    """
    session_id = request.session_id or str(uuid.uuid4())

    async with async_session_maker() as db:
        await save_message(db, session_id, "user", request.message)

        graph = build_graph(db)
        config = {"configurable": {"thread_id": session_id}}

        existing_state = await graph.aget_state(config)
        is_paused = bool(existing_state and existing_state.next)

        if is_paused:
            pending = _pending_interrupt_type(existing_state)
            affirmative = _is_affirmative(request.message)

            if pending == "confirm_trip_details":
                resume_value = (
                    {"confirmed": True}
                    if affirmative
                    else {"confirmed": False, "message": request.message}
                )
            elif pending == "confirm_generation_issues":
                resume_value = (
                    {"proceed": True}
                    if affirmative
                    else {"proceed": False, "message": request.message}
                )
            else:
                resume_value = {"confirmed": False, "message": request.message}

            result = await graph.ainvoke(Command(resume=resume_value), config=config)
        else:
            state = {
                "session_id": session_id,
                "user_id": request.user_id,
                "last_user_message": request.message,
            }
            result = await graph.ainvoke(state, config=config)

        response = _extract_response(session_id, request.user_id, result)
        if not response.interrupt:
            content_to_save = response.reply if response.reply else str(response.itinerary)
            await save_message(db, session_id, "assistant", content_to_save)

    return response


@app.get("/trips")
async def list_trips(user_id: str):
    """Backs the history panel's 'Your trips' list (app.js loadHistoryList)."""
    async with async_session_maker() as db:
        trips = await list_trips_for_user(db, user_id)
    return {"trips": trips}


@app.get("/session/{session_id}")
@app.get("/trip/{session_id}")
async def get_session(session_id: str, user_id: str):
    """Backs re-opening a past trip from history (app.js loadSession)."""
    async with async_session_maker() as db:
        trip_state, status = await load_trip(db, session_id, user_id)
        itinerary = await get_itinerary(db, session_id, user_id)
        history = await get_history(db, session_id)

    messages = [{"role": m["role"], "text": m["content"]} for m in history]
    return {
        "session_id": session_id,
        "user_id": user_id,
        "trip_state": trip_state or None,
        "itinerary": itinerary,
        "status": status,
        "messages": messages,
    }


@app.get("/places")
async def places(destination: str):
    """Categorized 'places to visit' guide for a destination (religious
    sites, waterfalls, wildlife, museums, unique experiences, etc), backed by
    a live web search. Independent of trip planning -- just an explore/lookup
    feature the frontend can call directly from a place-name search box."""
    guide = await get_places_guide(destination)
    return guide.model_dump()


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Lets you start the server with `python app/main.py` instead of typing
    # the uvicorn command by hand. --reload still works via the CLI if you
    # prefer that for development: `uvicorn app.main:app --reload`.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)