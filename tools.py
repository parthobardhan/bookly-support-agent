"""
LangChain tools for the Bookly support agent (MongoDB + policy vector search).
Tool outputs are intentionally compact JSON strings to limit context growth.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

from bson import json_util
from langchain_core.tools import tool
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo.collection import Collection

import config
import db
from voyage_embeddings import VoyageAIEmbeddings

# Hard cap on characters returned to the LLM per tool call
TOOL_JSON_MAX_CHARS = 2200

POLICY_TYPE_ALIASES: Dict[str, str] = {
    "shipping": "shipping",
    "return": "return",
    "password-reset": "password-reset",
}


def normalize_order_id(raw: str) -> str:
    """
    Map spoken or messy order ids to stored format (e.g. ORD M001 -> ORD-M001, ORD-5001).
    """
    s = raw.strip()
    if not s:
        return s
    compact = re.sub(r"\s+", "", s.upper())
    # ORDM001 / ORD-M001 style (letter suffix)
    m = re.match(r"^ORD-?M0*(\d+)$", compact)
    if m:
        return f"ORD-M{m.group(1)}"
    # ORD5001 / ORD-5001 (numeric only)
    m = re.match(r"^ORD-?(\d+)$", compact)
    if m:
        return f"ORD-{m.group(1)}"
    return s


def _truncate_chars(text: str, max_chars: int = TOOL_JSON_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]"


def _serialize_for_tool(value: Any) -> str:
    """Serialize MongoDB docs to JSON (ObjectId/datetime safe), then truncate."""
    raw = json_util.dumps(value, default=json_util.default)
    return _truncate_chars(raw)


def _policy_pre_filter(policy_type_key: str) -> Dict[str, Any]:
    """
    MQL pre-filter for $vectorSearch (filter-indexed string field).

    Must use MongoDB Query Language match syntax on indexed filter paths — not
    Atlas Search operators like ``equals`` / ``path`` / ``value`` (those apply
    to ``$search``, not ``$vectorSearch``).
    """
    ptype = POLICY_TYPE_ALIASES.get(policy_type_key)
    if not ptype:
        raise ValueError(
            f"Invalid policy type '{policy_type_key}'. "
            "Use: shipping, return, or password-reset."
        )
    return {"policy_type": ptype}


_vectorstore: MongoDBAtlasVectorSearch | None = None


def _get_policy_vectorstore() -> MongoDBAtlasVectorSearch:
    global _vectorstore
    if _vectorstore is None:
        coll = db.get_db()[config.POLICIES_COLLECTION]
        embeddings = VoyageAIEmbeddings()
        _vectorstore = MongoDBAtlasVectorSearch(
            collection=coll,
            embedding=embeddings,
            index_name=config.VECTOR_INDEX_NAME,
            text_key="text",
            embedding_key="embedding",
            relevance_score_fn="cosine",
            dimensions=config.VOYAGE_EMBED_DIMENSIONS,
            auto_create_index=False,
        )
    return _vectorstore


@tool
def lookup_order_status(order_id: str) -> str:
    """Look up one order by order_id. Returns all stored fields as JSON."""
    coll: Collection = db.get_db()[config.ORDERS_COLLECTION]
    oid = normalize_order_id(order_id)
    doc = coll.find_one({"order_id": oid})
    if not doc:
        return json.dumps({"found": False, "order_id": oid})
    return _serialize_for_tool({"found": True, "order": doc})


@tool
def list_recent_orders(
    customer_name: str,
    phone_number: Annotated[Optional[str], "Optional phone to disambiguate"] = None,
) -> str:
    """List recent orders for a customer name (newest first), optional phone filter."""
    coll: Collection = db.get_db()[config.ORDERS_COLLECTION]
    q: Dict[str, Any] = {"customer_name": customer_name.strip()}
    if phone_number:
        q["phone_number"] = phone_number.strip()
    cursor = coll.find(q).sort("order_id", -1).limit(8)
    orders = list(cursor)
    return _serialize_for_tool({"count": len(orders), "orders": orders})


@tool
def get_member_details(
    customer_name: str,
    dob: Annotated[Optional[str], "Optional date of birth as YYYY-MM-DD"] = None,
    phone_number: Optional[str] = None,
) -> str:
    """Look up a customer by name; optional DOB (YYYY-MM-DD) and phone to narrow results."""
    coll: Collection = db.get_db()[config.CUSTOMER_COLLECTION]
    q: Dict[str, Any] = {"customer_name": customer_name.strip()}
    if phone_number:
        q["phone_number"] = phone_number.strip()
    if dob:
        try:
            y, m, d = (int(x) for x in dob.strip().split("-"))
            dt = datetime(y, m, d, tzinfo=timezone.utc)
            q["DOB"] = dt
        except ValueError:
            return json.dumps({"error": "dob must be YYYY-MM-DD"})
    doc = coll.find_one(q)
    if not doc:
        return json.dumps({"found": False, "customer_name": customer_name})
    # Per product decision: include password for this demo (not recommended in production).
    return _serialize_for_tool({"found": True, "customer": doc})


@tool
def process_refund(
    order_id: str,
    book_name: str,
    book_type: str,
) -> str:
    """
    Record a return/replacement: set replacement book, mark return state,
    move order to Order Processing with a new estimated delivery date.
    """
    allowed = {"paperback", "ebook", "audiobook", "hardcover"}
    bt = book_type.strip().lower()
    if bt not in allowed:
        return json.dumps({"error": f"book_type must be one of {sorted(allowed)}"})
    coll: Collection = db.get_db()[config.ORDERS_COLLECTION]
    eta = datetime.now(timezone.utc) + timedelta(days=7)
    result = coll.update_one(
        {"order_id": order_id.strip()},
        {
            "$set": {
                "book_name": book_name.strip(),
                "book_type": bt,
                "status": "Order Processing",
                "return_replacement_state": "Replaced/Returned",
                "estimated_delivery_date": eta,
                "delivery_date": None,
            }
        },
    )
    updated = coll.find_one({"order_id": order_id.strip()})
    payload = {
        "matched": result.matched_count,
        "modified": result.modified_count,
        "order": updated,
    }
    return _serialize_for_tool(payload)


@tool
def search_policies(
    type: Literal["shipping", "return", "password-reset"],
    query: str,
) -> str:
    """
    Semantic search over Bookly policies. Always filters to one policy family:
    shipping, return, or password-reset.
    """
    vs = _get_policy_vectorstore()
    pre = _policy_pre_filter(type)
    docs = vs.similarity_search(
        query.strip(),
        k=3,
        pre_filter=pre,
    )
    chunks: List[Dict[str, Any]] = []
    for d in docs:
        chunks.append(
            {
                "text": d.page_content,
                "metadata": d.metadata,
            }
        )
    return _truncate_chars(json.dumps({"policy_type": type, "chunks": chunks}))


def all_tools() -> List:
    return [
        lookup_order_status,
        list_recent_orders,
        get_member_details,
        process_refund,
        search_policies,
    ]
