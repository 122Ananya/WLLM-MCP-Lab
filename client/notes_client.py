"""MCP client for the personal-assistant notes server.

Connects to server/notes_server.py over stdio, lets the user type a query,
and uses an LLM (Groq, OpenAI-compatible function calling) to decide whether
to call save_note or search_notes.
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Windows consoles default to a legacy codepage (e.g. cp1252) that can't
# encode characters like narrow no-break spaces that some models emit.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server", "notes_server.py")
MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are a personal assistant with access to a notes store. "
    "Use save_note when the user is telling you something to remember. "
    "Use search_notes when the user is asking a question about something "
    "they said before. Reply conversationally after using a tool."
)


def mcp_tools_to_openai(mcp_tools) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


async def run_chat():
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    server_params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            tools = mcp_tools_to_openai(tools_result.tools)

            print("Connected to notes MCP server. Type 'quit' to exit.\n")

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            while True:
                user_input = input("You: ").strip()
                if user_input.lower() in ("quit", "exit"):
                    break
                if not user_input:
                    continue

                messages.append({"role": "user", "content": user_input})

                while True:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                    msg = response.choices[0].message
                    tool_calls = msg.tool_calls or []

                    if not tool_calls:
                        print(f"Assistant: {msg.content}")
                        messages.append({"role": "assistant", "content": msg.content})
                        break

                    messages.append({
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    })

                    for tc in tool_calls:
                        args = json.loads(tc.function.arguments or "{}")
                        print(f"[calling {tc.function.name}({json.dumps(args)})]")
                        result = await session.call_tool(tc.function.name, args)
                        result_text = "\n".join(
                            c.text for c in result.content if hasattr(c, "text")
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_text,
                        })


def main():
    if "GROQ_API_KEY" not in os.environ:
        print("Set GROQ_API_KEY (env var or .env file) before running.")
        return
    try:
        asyncio.run(run_chat())
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
