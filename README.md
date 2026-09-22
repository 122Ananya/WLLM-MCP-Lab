# Personal Assistant Memory (MCP)

A minimal MCP server exposing `save_note` and `search_notes` tools backed by a local
`notes.json`, plus a Python client that uses Groq (OpenAI-compatible function calling)
to decide which tool to call from natural-language input.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key (free, from https://console.groq.com/keys) to `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Run

Terminal client:

```bash
python client/notes_client.py
```

Example:

```
You: Remember that my project deadline is this Friday
You: What did I say about the deadline?
```

Or the chat-style web UI (Streamlit):

```bash
streamlit run client/notes_app.py
```

## Structure

- `server/notes_server.py` — MCP server with `save_note(content, tags)` and `search_notes(query)`, reading/writing `server/notes.json`.
- `client/notes_client.py` — terminal MCP client that connects to the server over stdio and uses Groq to route user queries to the right tool.
- `client/notes_app.py` — Streamlit chat UI wrapping the same MCP + Groq flow, with a sidebar listing saved notes.
