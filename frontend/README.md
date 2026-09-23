# Payanam Frontend (plain HTML/JS)

Two pages, no build step needed:

- **chat.html** — the chat interface. Talks to your FastAPI backend's `/chat` endpoint.
- **trip.html** — a rich "Trip" view (Voyageur-style design) showing the full itinerary once it's generated. Opened via the "View full trip" link that appears in chat after a trip is generated.

## How to run

1. Make sure your backend is running (e.g. `uvicorn app.main:app --reload` from your `backend` folder).
2. Just open `chat.html` directly in your browser (double-click it, or use a simple local server like `python -m http.server` in this folder and visit `http://localhost:8000/chat.html`).
3. First time only: click **Settings** in the chat header and set the backend API URL if it's not `http://127.0.0.1:8000` (e.g. change the port if your backend runs elsewhere).

## How it works

- `chat.html` stores a random `user_id` in the browser's localStorage the first time you use it (simulating a logged-in user). It also remembers your current `session_id` so follow-up messages continue the same conversation.
- When the backend returns an `interrupt` (a paused confirmation step), the chat shows two buttons matching whatever `interrupt.type` is: "Confirm trip" / "Change something" for trip details, or "Proceed anyway" / "Fix the issues" for validation problems. Clicking "Change something"/"Fix the issues" just focuses the input box so you can type your correction; clicking the confirm button sends "yes" automatically.
- Once an itinerary is generated, a short summary appears in chat with a **"View full trip →"** link. Clicking it opens `trip.html?session_id=...&user_id=...`, which calls the backend's `GET /trip/{session_id}?user_id=...` endpoint and renders the full day-by-day itinerary in a nicer layout.
- Click **"New trip"** in the chat header to clear the current session and start a fresh conversation (a new `session_id` will be created on your next message).

## Backend requirement

This frontend expects your `app/main.py` to have the `GET /trip/{session_id}` endpoint (returns `trip_state`, `status`, `itinerary`) in addition to the existing `POST /chat` endpoint. If you haven't added it yet, add this to `app/main.py`:

```python
@app.get("/trip/{session_id}")
async def get_trip(session_id: str, user_id: str):
    async with async_session_maker() as db:
        trip_state, status = await load_trip(db, session_id, user_id)
        itinerary = await get_itinerary(db, session_id, user_id)
        return {
            "session_id": session_id,
            "trip_state": trip_state,
            "status": status,
            "itinerary": itinerary,
        }
```

## Notes / things to know

- This is plain HTML/CSS/JS — no npm, no build tools, no frameworks. Easy to tweak directly.
- CORS: your backend needs `CORSMiddleware` enabled (already added in earlier steps) or the browser will block these requests.
- The Trip page only shows real data returned by your backend — it does not fabricate hotel photos, maps, or booking confirmations like the original design mockup did, since we don't have that data from the API yet.
- Everything is stored in browser localStorage only (`user_id`, `session_id`, API base URL) — no backend auth is wired up. Good enough for local testing, not for production multi-user use.
