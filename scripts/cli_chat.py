import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from infrastructure.config.composition_root import create_application

console = Console()


async def main():
    app = create_application()
    config: RunnableConfig = {
        "configurable": {"thread_id": "test-session", "user_id": "test_user"}
    }

    console.print()
    console.rule("[bold]AI Agent Chat")
    console.print()

    while True:
        user = Prompt.ask("[bold cyan]You")
        if user.lower() in ("exit", "quit"):
            break

        full_response = ""
        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            async for event in app.astream(
                {"messages": [HumanMessage(content=user)]},
                config,
                stream_mode="updates",
            ):
                if "assistant" not in event:
                    continue
                for msg in event["assistant"].get("messages", []):
                    if isinstance(msg, AIMessage) and msg.content:
                        full_response = str(msg.content)
        console.print()
        console.print(
            Panel(
                Markdown(full_response),
                title="[bold green]Agent[/bold green]",
                title_align="left",
                box=box.ROUNDED,
                border_style="green",
            )
        )
        console.print()


if __name__ == "__main__":
    asyncio.run(main())
