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
