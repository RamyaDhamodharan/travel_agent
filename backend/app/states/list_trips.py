def format_trip_list(trips: list[dict]) -> str:
    if not trips:
        return "You don't have any trips planned yet. Want to start one?"

    lines = ["Here are your trips:"]
    for i, t in enumerate(trips, start=1):
        dest = t.get("destination") or "Unnamed trip"
        dates = f"{t.get('start_date', '?')} to {t.get('end_date', '?')}"
        status = t.get("status", "unknown")
        lines.append(f"{i}. {dest} ({dates}) — status: {status}")
    return "\n".join(lines)
