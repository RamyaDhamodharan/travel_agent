from app.graph import build_graph

graph = build_graph(db=None)

# Save as PNG image (uses Mermaid's online rendering API - needs internet)
png_bytes = graph.get_graph().draw_mermaid_png()

with open("graph_diagram.png", "wb") as f:
    f.write(png_bytes)

print("Saved as graph_diagram.png")