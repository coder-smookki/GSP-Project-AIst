from pathlib import Path

from infrastructure.config.composition_root import create_application

app = create_application()

out = Path(__file__).resolve().parents[1] / "docs" / "demo" / "graph.png"
out.parent.mkdir(parents=True, exist_ok=True)

png = app.get_graph().draw_mermaid_png()
out.write_bytes(png)

print(f"OK: graph.png saved to {out}")
