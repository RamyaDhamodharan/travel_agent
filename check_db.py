import asyncio
from app.db.session import async_session_maker
from app.db.models import TripSession
from sqlalchemy import select


async def main():
    async with async_session_maker() as session:
        result = await session.execute(
            select(
                TripSession.id,
                TripSession.session_id,
                TripSession.user_id,
                TripSession.status,
                TripSession.trip_state,
                TripSession.created_at,
            )
            .where(TripSession.user_id == "test_user_1")
            .order_by(TripSession.created_at)
        )
        rows = result.all()
        print(f"Total rows: {len(rows)}")
        for r in rows:
            print(r)


asyncio.run(main())