from infrastructure.config.composition_root import create_application

app = create_application()

png = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png)

print("OK: graph.png saved")
