# API Sample Responses

Base URL: `http://localhost:8000`

## 1) Health Check
### Request
```http
GET /health
```

### Response (200)
```json
{
  "status": "ok"
}
```

## 2) Ingest Initial Memory Turn
### Request
```http
POST /sessions/demo/turns
Content-Type: application/json

{
  "user_message": "My preferred language is Kannada.",
  "expose_memory": true
}
```

### Response (200)
```json
{
  "session_id": "demo",
  "user_turn_id": 1,
  "assistant_turn_id": 2,
  "response_text": "Understood. I noted that and will use it in future responses.",
  "response_generated": true,
  "latency_ms": 2.731,
  "active_memories": []
}
```

Note: `active_memories` can be empty on the same turn because extraction is asynchronous.

## 3) Recall at Later Turn
### Request
```http
POST /sessions/demo/turns
Content-Type: application/json

{
  "user_message": "Can we continue in my language?",
  "expose_memory": true
}
```

### Response (200)
```json
{
  "session_id": "demo",
  "user_turn_id": 45,
  "assistant_turn_id": 46,
  "response_text": "We can continue in Kannada.",
  "response_generated": true,
  "latency_ms": 2.518,
  "active_memories": [
    {
      "memory_id": "mem_08c30ee7ef82",
      "content": "User prefers Kannada",
      "origin_turn": 1,
      "last_used_turn": 44,
      "key": "preferred_language",
      "value": "Kannada",
      "score": 0.7689
    }
  ]
}
```

## 4) List Memories for a Session
### Request
```http
GET /sessions/demo/memories
```

### Response (200)
```json
{
  "session_id": "demo",
  "memories": [
    {
      "memory_id": "mem_08c30ee7ef82",
      "type": "preference",
      "key": "preferred_language",
      "value": "Kannada",
      "source_turn": 1,
      "last_used_turn": 45,
      "confidence": 0.95,
      "status": "active",
      "supersedes_memory_id": null,
      "subject": "user",
      "relation": "prefers_language",
      "object": "Kannada",
      "score": null
    }
  ]
}
```

## 5) Flush Background Queue
### Request
```http
POST /sessions/demo/flush
```

### Response (200)
```json
{
  "session_id": "demo",
  "pending_tasks": 0
}
```

## 6) Run Built-in Evaluation
### Request
```http
POST /evaluate
```

### Response (200)
```json
{
  "recall_rate": 1.0,
  "retrieval_relevance": 1.0,
  "latency_p50_ms": 2.426,
  "latency_p95_ms": 3.483,
  "contradiction_rate": 0.0,
  "total_probes": 3
}
```

## Optional cURL snippets

```bash
curl -X POST "http://localhost:8000/sessions/demo/turns" \
  -H "Content-Type: application/json" \
  -d '{"user_message":"My preferred language is Kannada.","expose_memory":true}'

curl "http://localhost:8000/sessions/demo/memories"

curl -X POST "http://localhost:8000/evaluate"
```
