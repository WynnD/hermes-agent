# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search, reranking, and automatic deduplication. Supports both the hosted Mem0 Platform and a self-hosted Mem0 REST server.

## Requirements

- `pip install mem0ai`
- Hosted: Mem0 API key from [app.mem0.ai](https://app.mem0.ai)
- Self-hosted: Mem0 REST server URL

## Setup

```bash
hermes memory setup    # select "mem0"
```

Or manually:
```bash
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

Self-hosted example:

```json
{
  "base_url": "https://mem0.example.com",
  "user_id": "wynn",
  "agent_id": "hermes",
  "rerank": true
}
```

## Config

Config file: `$HERMES_HOME/mem0.json`

| Key | Default | Description |
|-----|---------|-------------|
| `api_key` | unset | Hosted Mem0 API key, or optional self-hosted `X-API-Key` |
| `base_url` | unset | Self-hosted Mem0 server URL. If set, no hosted API key is required. |
| `user_id` | `hermes-user` | User identifier on Mem0 |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `true` | Enable reranking for recall |

If `user_id` is configured, it is treated as the canonical memory scope across CLI and gateway surfaces. This avoids splitting a single user's memory between Discord IDs, Telegram IDs, and CLI defaults.

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | All stored memories about the user |
| `mem0_search` | Semantic search with optional reranking |
| `mem0_conclude` | Store a fact verbatim (no LLM extraction) |
