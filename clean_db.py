import asyncio
from app.db.session import async_session_maker
from app.db.models import TripSession
from sqlalchemy import delete


async def main():
    async with async_session_maker() as session:
        await session.execute(delete(TripSession).where(TripSession.user_id == "test_user_1"))
        await session.commit()
    print("Done - test_user_1 trips deleted.")


asyncio.run(main())