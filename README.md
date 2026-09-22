# Personal Assistant Memory (MCP)

A minimal MCP server exposing `save_note` and `search_notes` tools backed by a local
`notes.json`, plus a Python client that uses Gemini (function calling) to decide
which tool to call from natural-language input.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your Gemini API key to `.env`:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Run

```bash
python client/notes_client.py
```

Example:

```
You: Remember that my project deadline is this Friday
You: What did I say about the deadline?
```

## Structure

- `server/notes_server.py` — MCP server with `save_note(content, tags)` and `search_notes(query)`, reading/writing `server/notes.json`.
- `client/notes_client.py` — MCP client that connects to the server over stdio and uses Gemini to route user queries to the right tool.
