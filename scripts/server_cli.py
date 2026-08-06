import asyncio
import os
import sys

from langgraph_sdk import get_client
from langgraph_sdk.schema import Config
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

SERVER_URL = os.getenv("AEGRA_URL", "http://localhost:2026")
USER_ID = os.getenv("AEGRA_USER_ID", "anonymous")

RUN_CONFIG: Config = {
    "configurable": {
        "user_id": USER_ID,
    },
}

console = Console()


async def main():
    client = get_client(url=SERVER_URL)

    assistants = await client.assistants.search()
    if not assistants:
        console.print("[red]No assistants found on server[/red]")
        sys.exit(1)

    assistant_id = assistants[0]["assistant_id"]
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    console.print()
    console.rule("[bold]AI Agent Chat")
    console.print(f"  [dim]Server:[/dim]     {SERVER_URL}")
    console.print(f"  [dim]Assistant:[/dim] {assistant_id}")
    console.print(f"  [dim]Thread ID:[/dim] {thread_id[:8]}...")
    console.print(f"  [dim]User ID:[/dim]   {USER_ID}")
    console.print()

    while True:
        user = Prompt.ask("[bold cyan]You")
        if user.lower() in ("exit", "quit"):
            break

        full_response = ""

        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            async for chunk in client.runs.stream(
                thread_id=thread_id,
                assistant_id=assistant_id,
                input={"messages": [{"type": "human", "content": user}]},
                config=RUN_CONFIG,
                stream_mode=["updates"],
            ):
                if chunk.event != "updates" or not chunk.data:
                    continue
                for node, data in chunk.data.items():
                    if node == "assistant":
                        for msg in data.get("messages") or []:
                            if msg.get("type") == "ai" and msg.get("content"):
                                full_response = msg["content"]

        console.print()

        console.print(
            Panel(
                Markdown(full_response or "*[dim]No response[/dim]*"),
                title="[bold green]Agent[/bold green]",
                title_align="left",
                box=box.ROUNDED,
                border_style="green",
            )
        )
        console.print()

    console.print("[yellow]Goodbye![/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
