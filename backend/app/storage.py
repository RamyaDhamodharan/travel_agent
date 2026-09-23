from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TripSession, Message


async def load_trip(db: AsyncSession, session_id: str, user_id: str) -> tuple[dict, str]:
    """Returns (trip_state, status). status='new' if no row exists yet,
    or if the row exists but belongs to a DIFFERENT user_id -- this is
    the security guardrail preventing session_id hijacking across users."""
    result = await db.execute(select(TripSession).where(TripSession.session_id == session_id))
    row = result.scalar_one_or_none()
    if row is None or row.user_id != user_id:
        return {}, "new"
    return row.trip_state or {}, row.status


async def save_trip(
    db: AsyncSession,
    session_id: str,
    user_id: str,
    trip_state: dict,
    itinerary: dict | None = None,
    summary: str | None = None,
    status: str = "collecting",
    validation_issues: list[str] | None = None,
) -> None:
    if validation_issues is not None:
        trip_state = {**trip_state, "validation_issues": validation_issues}

    result = await db.execute(select(TripSession).where(TripSession.session_id == session_id))
    row = result.scalar_one_or_none()

    if row is None:
        row = TripSession(
            session_id=session_id,
            user_id=user_id,
            trip_state=trip_state,
            itinerary=itinerary,
            summary=summary,
            status=status,
        )
        db.add(row)
    else:
        row.trip_state = trip_state
        if itinerary is not None:
            row.itinerary = itinerary
        if summary is not None:
            row.summary = summary
        row.status = status

    await db.commit()


async def get_itinerary(db: AsyncSession, session_id: str, user_id: str) -> dict | None:
    result = await db.execute(select(TripSession).where(TripSession.session_id == session_id))
    row = result.scalar_one_or_none()
    if row is None or row.user_id != user_id:
        return None
    return row.itinerary


async def list_trips_for_user(db: AsyncSession, user_id: str, exclude_session_id: str | None = None, limit: int = 20) -> list[dict]:
    """Lightweight summaries only — no full itinerary JSON. Cheap for context window."""
    query = select(TripSession).where(TripSession.user_id == user_id).order_by(TripSession.updated_at.desc()).limit(limit)
    result = await db.execute(query)
    rows = result.scalars().all()
    trips = []
    for r in rows:
        if exclude_session_id and r.session_id == exclude_session_id:
            continue
        state = r.trip_state or {}
        trips.append(
            {
                "session_id": r.session_id,
                "destination": state.get("destination"),
                "start_date": state.get("start_date"),
                "end_date": state.get("end_date"),
                "status": r.status,
                "summary": r.summary,
            }
        )
    return trips


async def find_trip_by_destination(db: AsyncSession, user_id: str, destination_hint: str) -> dict | None:
    """Case-insensitive fuzzy match on destination within a user's trips."""
    result = await db.execute(select(TripSession).where(TripSession.user_id == user_id))
    rows = result.scalars().all()
    hint = destination_hint.lower().strip()
    for r in rows:
        dest = (r.trip_state or {}).get("destination", "")
        if dest and (hint in dest.lower() or dest.lower() in hint):
            return {
                "session_id": r.session_id,
                "trip_state": r.trip_state,
                "itinerary": r.itinerary,
                "summary": r.summary,
                "status": r.status,
            }
    return None


async def save_message(db: AsyncSession, session_id: str, role: str, content: str) -> None:
    db.add(Message(session_id=session_id, role=role, content=content))
    await db.commit()


async def get_history(db: AsyncSession, session_id: str, limit: int = 20) -> list[dict]:
    result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at).limit(limit)
    )
    return [{"role": m.role, "content": m.content} for m in result.scalars().all()]