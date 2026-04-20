"""MongoDB client and database access."""

from __future__ import annotations

import os

import certifi
from pymongo import MongoClient

import config

# Singleton client: reuse one MongoClient per process (thread-safe, connection-pooled)
_client: MongoClient | None = None


def get_mongodb_uri() -> str:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set.")
    return uri


def get_client() -> MongoClient:
    """Return MongoDB client with certifi CA bundle for SSL verification. Reuses a single client per process."""
    global _client
    if _client is None:
        _client = MongoClient(get_mongodb_uri(), tlsCAFile=certifi.where())
    return _client


def get_db():
    """Return the Bookly application database."""
    return get_client()[config.MONGODB_DB_NAME]
