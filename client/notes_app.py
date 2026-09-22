"""Streamlit chat UI for the personal-assistant notes MCP server.

Same MCP server + Groq tool-calling flow as notes_client.py, just with a
browser-based chat UI instead of a terminal loop. Run with:

    streamlit run client/notes_app.py
"""

import asyncio
import json
import os
import sys

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from notes_client import MODEL, SERVER_SCRIPT, SYSTEM_PROMPT, mcp_tools_to_openai

load_dotenv()

NOTES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server", "notes.json")

st.set_page_config(page_title="Personal Assistant", page_icon="🧠", layout="wide")


def load_notes() -> list:
    if not os.path.exists(NOTES_PATH):
        return []
    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


async def ask(messages: list, groq_client: Groq) -> tuple[list, list]:
    """Run one full MCP round-trip (possibly several tool calls) for the
    latest user message. Returns (updated messages, tool_call_log)."""
    server_params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])
    tool_log = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            tools = mcp_tools_to_openai(tools_result.tools)

            while True:
                response = groq_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                msg = response.choices[0].message
                tool_calls = msg.tool_calls or []

                if not tool_calls:
                    messages.append({"role": "assistant", "content": msg.content})
                    return messages, tool_log

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
                    result = await session.call_tool(tc.function.name, args)
                    result_text = "\n".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )
                    tool_log.append({"name": tc.function.name, "args": args, "result": result_text})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })


st.title("🧠 Personal Assistant")
st.caption("Backed by an MCP server (`save_note` / `search_notes`) — routed by an LLM via Groq.")

if "GROQ_API_KEY" not in os.environ:
    st.error("Set GROQ_API_KEY in your .env file (see README), then restart Streamlit.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "display" not in st.session_state:
    st.session_state.display = []

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

with st.sidebar:
    st.subheader("📒 Saved notes")
    if st.button("🔄 Refresh"):
        st.rerun()

    notes = load_notes()
    if not notes:
        st.caption("No notes saved yet.")
    for note in reversed(notes):
        with st.container(border=True):
            st.write(note["content"])
            if note.get("tags"):
                st.caption(" ".join(f"`{t}`" for t in note["tags"]))
            st.caption(note["created_at"])

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.display = []
        st.rerun()

for turn in st.session_state.display:
    with st.chat_message(turn["role"]):
        for call in turn.get("tool_log", []):
            st.info(f"🔧 called **{call['name']}**({json.dumps(call['args'])})")
        st.write(turn["content"])

user_input = st.chat_input("Tell me something to remember, or ask what you told me...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.display.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        updated_messages, tool_log = asyncio.run(ask(st.session_state.messages, groq_client))

    st.session_state.messages = updated_messages
    final_text = updated_messages[-1]["content"]
    st.session_state.display.append({"role": "assistant", "content": final_text, "tool_log": tool_log})
    st.rerun()
