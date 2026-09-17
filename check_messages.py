import asyncio
from app.db.session import async_session_maker
from app.db.models import Message
from sqlalchemy import select


async def main():
    async with async_session_maker() as session:
        result = await session.execute(
            select(Message.session_id, Message.role, Message.content, Message.created_at)
            .order_by(Message.created_at)
        )
        rows = result.all()
        print(f"Total message rows: {len(rows)}")
        for r in rows:
            print(r)


asyncio.run(main())