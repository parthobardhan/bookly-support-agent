"""
Shared configuration for Bookly demo app (MongoDB, vector index, model IDs).
"""

import os

# MongoDB database and collection names
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "bookly")
CUSTOMER_COLLECTION = "customer"
ORDERS_COLLECTION = "orders"
POLICIES_COLLECTION = "policies"
CHECKPOINT_COLLECTION = "checkpoints"

# Atlas Vector Search index name (must match README / Atlas UI)
VECTOR_INDEX_NAME = os.getenv("MONGODB_VECTOR_INDEX_NAME", "bookly_policy_vector_index")

# Voyage (default dimension for voyage-4 at default output is 1024)
VOYAGE_EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-4")
VOYAGE_EMBED_DIMENSIONS = int(os.getenv("VOYAGE_EMBED_DIMENSIONS", "1024"))

# OpenAI chat model (set OPENAI_MODEL to your available chat model)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Deepgram — batch STT (fallback) and Aura TTS model name (REST + Speak WS)
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-2")
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")

# Flux Listen v2 (streaming STT with end-of-turn)
DEEPGRAM_FLUX_MODEL = os.getenv("DEEPGRAM_FLUX_MODEL", "flux-general-en")
FLUX_EOT_THRESHOLD = os.getenv("FLUX_EOT_THRESHOLD", "0.7")
FLUX_EOT_TIMEOUT_MS = int(os.getenv("FLUX_EOT_TIMEOUT_MS", "5000"))
_eager = os.getenv("FLUX_EAGER_EOT_THRESHOLD")
FLUX_EAGER_EOT_THRESHOLD = _eager if _eager else None

# Speak v1 WebSocket (streaming TTS) — linear16; packaged as WAV for st.audio
DEEPGRAM_SPEAK_SAMPLE_RATE = int(os.getenv("DEEPGRAM_SPEAK_SAMPLE_RATE", "24000"))
