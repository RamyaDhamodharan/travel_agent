"""
Entry point for LangGraph Studio only. Studio needs a graph object (or a
factory that returns one) that it can import directly -- it doesn't know
about our per-request db session from main.py. This creates ONE db session
that stays open for the whole Studio debugging session and builds the graph
against it.
"""
from app.db.session import async_session_maker
from app.graph import build_graph

_db_session = None
_graph = None


async def get_graph():
    global _db_session, _graph
    if _graph is None:
        _db_session = async_session_maker()
        _db_session = await _db_session.__aenter__()
        _graph = build_graph(_db_session)
    return _graph