"""MCP client for the personal-assistant notes server.

Connects to server/notes_server.py over stdio, lets the user type a query,
and uses an LLM (Gemini) with function-calling to decide whether to call
save_note or search_notes.
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server", "notes_server.py")
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a personal assistant with access to a notes store. "
    "Use save_note when the user is telling you something to remember. "
    "Use search_notes when the user is asking a question about something "
    "they said before. Reply conversationally after using a tool."
)


def mcp_schema_to_gemini(schema: dict) -> dict:
    """Strip fields Gemini's schema doesn't accept (e.g. 'title', 'default')."""
    if not isinstance(schema, dict):
        return schema
    allowed_keys = {"type", "description", "properties", "items", "required", "enum"}
    cleaned = {k: v for k, v in schema.items() if k in allowed_keys}
    if "properties" in cleaned:
        cleaned["properties"] = {
            k: mcp_schema_to_gemini(v) for k, v in cleaned["properties"].items()
        }
    if "items" in cleaned:
        cleaned["items"] = mcp_schema_to_gemini(cleaned["items"])
    return cleaned


def mcp_tools_to_gemini(mcp_tools) -> genai_types.Tool:
    declarations = [
        genai_types.FunctionDeclaration(
            name=t.name,
            description=t.description or "",
            parameters=mcp_schema_to_gemini(t.inputSchema),
        )
        for t in mcp_tools
    ]
    return genai_types.Tool(function_declarations=declarations)


async def run_chat():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    server_params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            gemini_tool = mcp_tools_to_gemini(tools_result.tools)

            print("Connected to notes MCP server. Type 'quit' to exit.\n")

            chat = client.chats.create(
                model=MODEL,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[gemini_tool],
                ),
            )

            while True:
                user_input = input("You: ").strip()
                if user_input.lower() in ("quit", "exit"):
                    break
                if not user_input:
                    continue

                response = chat.send_message(user_input)

                while True:
                    function_calls = [
                        part.function_call
                        for part in response.candidates[0].content.parts
                        if part.function_call
                    ]
                    if not function_calls:
                        print(f"Assistant: {response.text}")
                        break

                    function_responses = []
                    for call in function_calls:
                        args = dict(call.args) if call.args else {}
                        print(f"[calling {call.name}({json.dumps(args)})]")
                        result = await session.call_tool(call.name, args)
                        result_text = "\n".join(
                            c.text for c in result.content if hasattr(c, "text")
                        )
                        function_responses.append(
                            genai_types.Part.from_function_response(
                                name=call.name,
                                response={"result": result_text},
                            )
                        )
                    response = chat.send_message(function_responses)


def main():
    if "GEMINI_API_KEY" not in os.environ:
        print("Set GEMINI_API_KEY (env var or .env file) before running.")
        return
    try:
        asyncio.run(run_chat())
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
