import asyncio
import uuid

from app.db.session import async_session_maker
from app.graph import build_graph
from app.storage import save_message


async def chat_loop():
    session_id = str(uuid.uuid4())
    print(f"New session: {session_id}")
    print("Trip planning agent ready. Type your message (or 'quit' to exit).\n")

    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in ("quit", "exit"):
            break

        async with async_session_maker() as db:
            await save_message(db, session_id, "user", user_message)

            graph = build_graph(db)
            state = {
                "session_id": session_id,
                "user_id": session_id,
                "last_user_message": user_message,
            }
            result = await graph.ainvoke(state)

            reply = result.get("assistant_reply", "")
            if reply:
                print(f"\nAssistant: {reply}\n")
                await save_message(db, session_id, "assistant", reply)
            elif result.get("itinerary"):
                print(f"\nAssistant: Here's your itinerary:\n{result['itinerary']}\n")
                await save_message(db, session_id, "assistant", str(result["itinerary"]))


if __name__ == "__main__":
    asyncio.run(chat_loop())