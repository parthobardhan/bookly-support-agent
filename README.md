# Bookly customer support agent (demo)

Structured Python demo for a conversational support agent for the fictional bookstore **Bookly**. Stack: LangGraph, OpenAI chat, Voyage AI embeddings, MongoDB Atlas (data + vector store + LangGraph checkpoints), Deepgram (Nova STT, Aura TTS), Arize Phoenix tracing (via OpenInference), Streamlit UI.

## Prerequisites

- Python 3.11 or newer recommended
- MongoDB Atlas cluster (M10+ recommended for Atlas Search; vector search indexes are not available on M0 in some accounts; check your Atlas tier)
- API keys: OpenAI, Voyage AI, Deepgram

## Setup

1. Clone or copy this project and create a virtual environment:

```bash
cd bookly-support-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Copy `.env.example` to `.env` and set `MONGODB_URI`, `OPENAI_API_KEY`, `VOYAGE_API_KEY`, and `DEEPGRAM_API_KEY`.
2. Create the **Atlas Vector Search** index on the `policies` collection **before** or **after** seeding (name must match `MONGODB_VECTOR_INDEX_NAME`, default `bookly_policy_vector_index`).

### Vector index definition

Create a vector search index in Atlas on database `bookly`, collection `policies`, with JSON similar to:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "policy_type"
    }
  ]
}
```

Notes:

- `numDimensions` **1024** matches the default output size for `voyage-4` (see Voyage docs). If you change `output_dimension` in code or env, update the index to match.
- The **filter** on `policy_type` is required for pre-filtered semantic search (`shipping`, `return`, `password-reset`) in `search_policies`.

1. Seed mock data and policy embeddings:

```bash
python database_seed.py
```

This clears and repopulates `customer`, `orders`, and `policies`. It uses `MongoDBAtlasVectorSearch` with `auto_create_index=True` to attempt index creation; if your cluster cannot auto-create indexes, create the index manually as above and re-run if needed.

1. Run the app:

```bash
streamlit run app.py
```

## Tracing (Arize Phoenix)

The legacy import `phoenix.trace.langchain.LangChainInstrumentor` was removed in Phoenix 4.x. This project uses the supported pattern:

- `phoenix.otel.register()` for the tracer provider
- `openinference.instrumentation.langchain.LangChainInstrumentor`

Spans export to the collector shown in the console (default gRPC `localhost:4317`). Run Phoenix locally or point OTEL env vars at your deployment per [Phoenix docs](https://arize.com/docs/phoenix/tracing/integrations-tracing/langchain).

## Data model (seeded)

- **customer**: nested `preferences.book_format` (`paperback`, `ebook`, `audiobook`, `hardcover`), embedded `shipping` address object, plus `customer_id`, `customer_name`, `password` (mock), `DOB`, `phone_number`.
- **orders**: embedded `shipping`; `status` includes `Order Processing` and `In Transit` with `estimated_delivery_date`; `Delivered` orders use `delivery_date` instead.
- **policies**: chunked text with `policy_type` and Voyage embeddings in `embedding` for vector search.

At least one customer (**Alice Reader**) has orders in `orders`.

## Example flows (input and expected behavior)

### 1. Multi-turn clarification (order ID)

**User:** Where is my package?

**Expected assistant:** Asks for an **order ID** (and stays concise) before calling `lookup_order_status`.

**User:** ORD-5001

**Expected:** Calls `lookup_order_status`, summarizes status (for example In Transit) and ETA in plain language.

### 2. Tool-backed action (replacement / return flow)

**User:** I want to return ORD-5002 and get "City of Leaves" as a hardcover instead.

**Expected:** Asks clarifying questions if needed (replacement title and format, address on file vs new), may call `get_member_details`, then calls `process_refund` so the order moves to **Order Processing**, `return_replacement_state` **Replaced/Returned**, with a new **estimated_delivery_date**.

### 3. Policy RAG with pre-filter

**User:** How long do I have to start a return?

**Expected:** Calls `search_policies` with `type="return"` and a short query; answer reflects the seeded rule that returns or refunds should be **initiated within 30 days of the delivery date**.

## Troubleshooting

- If you see `ImportError: cannot import name 'SON' from 'bson'`, uninstall the standalone PyPI package named `bson` (`pip uninstall bson`). PyMongo ships its own compatible `bson` module; the external package conflicts with it.

## Security note

`get_member_details` returns the mock `password` field on purpose for this lab demo. Do not enable this in production.

## Project layout


| File                   | Role                                                              |
| ---------------------- | ----------------------------------------------------------------- |
| `config.py`            | DB names, index name, model defaults                              |
| `voyage_embeddings.py` | LangChain `Embeddings` wrapper over `voyageai.Client`             |
| `database_seed.py`     | Customers, orders, policy chunks + embeddings                     |
| `tools.py`             | LangChain tools (orders, customers, refund update, policy search) |
| `audio_handler.py`     | Deepgram REST STT/TTS                                             |
| `agent.py`             | LangGraph + Phoenix instrumentation + MongoDB checkpointer        |
| `app.py`               | Streamlit text/voice UI                                           |


