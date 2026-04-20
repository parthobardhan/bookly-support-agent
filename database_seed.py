"""
Seed MongoDB Atlas with mock customers, orders, and policy chunks + embeddings.

Run once after creating the vector search index (see README), or let auto_create_index
attempt creation if your cluster allows it.

Usage:
    python database_seed.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

import config
import db
from voyage_embeddings import VoyageAIEmbeddings

load_dotenv()

# Policy texts (chunked manually for the demo)
SHIPPING_POLICY_CHUNKS = [
    (
        "Bookly ships physical books within the United States within 3 business days "
        "of payment confirmation. Digital orders (ebook and audiobook) are delivered "
        "to your Bookly library immediately after checkout."
    ),
    (
        "International shipping may take 7 to 21 business days depending on customs. "
        "You will receive tracking information by email when your package ships. "
        "Shipping fees are calculated at checkout based on weight and destination."
    ),
]

RETURN_POLICY_CHUNKS = [
    (
        "Bookly accepts returns of eligible physical books within 30 days of the "
        "delivery date shown on your order. To qualify, items must be unused and in "
        "original packaging unless the item arrived damaged or incorrect."
    ),
    (
        "Returns or refunds should be initiated within 30 days of the delivery date. "
        "Digital purchases (ebook and audiobook) are generally non-refundable once "
        "downloaded, except where required by law or if the file is defective."
    ),
    (
        "To start a return, contact Bookly support with your order ID. Refunds are "
        "processed to the original payment method within 5 to 10 business days after "
        "we receive the returned item at our warehouse."
    ),
]

PASSWORD_RESET_CHUNKS = [
    (
        "To reset your Bookly account password, open the Bookly website, click "
        "'Sign in', then choose 'Forgot password'. Enter the email address on your "
        "account and submit the form."
    ),
    (
        "You will receive a password reset link by email. The link expires after "
        "one hour for security. If you do not see the email, check spam and verify "
        "you entered the correct address. For repeated issues, contact support."
    ),
]


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _seed_customers(collection) -> None:
    customers: List[Dict[str, Any]] = [
        {
            "customer_id": "CUST-001",
            "customer_name": "Alice Reader",
            "password": "alice-mock-secret",
            "DOB": datetime(1990, 5, 15, tzinfo=timezone.utc),
            "phone_number": "+1-555-0101",
            "preferences": {"book_format": "paperback"},
            "shipping": {
                "line1": "123 Book St",
                "line2": "Apt 4",
                "city": "Seattle",
                "region": "WA",
                "postal_code": "98101",
                "country": "US",
            },
        },
        {
            "customer_id": "CUST-002",
            "customer_name": "Ben Audiophile",
            "password": "ben-mock-secret",
            "DOB": datetime(1988, 11, 2, tzinfo=timezone.utc),
            "phone_number": "+1-555-0102",
            "preferences": {"book_format": "audiobook"},
            "shipping": {
                "line1": "88 Sound Ave",
                "line2": "",
                "city": "Austin",
                "region": "TX",
                "postal_code": "78701",
                "country": "US",
            },
        },
        {
            "customer_id": "CUST-003",
            "customer_name": "Clara Hardcover",
            "password": "clara-mock-secret",
            "DOB": datetime(1995, 1, 20, tzinfo=timezone.utc),
            "phone_number": "+1-555-0103",
            "preferences": {"book_format": "hardcover"},
            "shipping": {
                "line1": "400 Library Ln",
                "line2": "",
                "city": "Boston",
                "region": "MA",
                "postal_code": "02108",
                "country": "US",
            },
        },
        {
            "customer_id": "CUST-004",
            "customer_name": "Dev Ebook",
            "password": "dev-mock-secret",
            "DOB": datetime(1992, 7, 8, tzinfo=timezone.utc),
            "phone_number": "+1-555-0104",
            "preferences": {"book_format": "ebook"},
            "shipping": {
                "line1": "9 Pixel Rd",
                "line2": "Suite 200",
                "city": "Denver",
                "region": "CO",
                "postal_code": "80202",
                "country": "US",
            },
        },
        {
            "customer_id": "CUST-005",
            "customer_name": "Eve Mixed",
            "password": "eve-mock-secret",
            "DOB": datetime(1984, 3, 30, tzinfo=timezone.utc),
            "phone_number": "+1-555-0105",
            "preferences": {"book_format": "paperback"},
            "shipping": {
                "line1": "77 Chapter Ct",
                "line2": "",
                "city": "Portland",
                "region": "OR",
                "postal_code": "97201",
                "country": "US",
            },
        },
    ]
    collection.delete_many({})
    collection.insert_many(customers)
    print(f"Inserted {len(customers)} customers.")


def _seed_orders(collection) -> None:
    today = datetime.now(timezone.utc).date()
    d1 = datetime.combine(today + timedelta(days=2), datetime.min.time(), tzinfo=timezone.utc)
    d2 = datetime.combine(today + timedelta(days=5), datetime.min.time(), tzinfo=timezone.utc)
    d_delivered = datetime.combine(today - timedelta(days=10), datetime.min.time(), tzinfo=timezone.utc)

    def ship(addr_name: str) -> Dict[str, Any]:
        return {
            "line1": "123 Book St",
            "line2": "Apt 4",
            "city": "Seattle",
            "region": "WA",
            "postal_code": "98101",
            "country": "US",
            "label_name": addr_name,
        }

    orders: List[Dict[str, Any]] = [
        {
            "order_id": "ORD-5001",
            "customer_name": "Alice Reader",
            "phone_number": "+1-555-0101",
            "book_name": "The Quiet Atlas",
            "book_type": "paperback",
            "status": "In Transit",
            "estimated_delivery_date": d1,
            "delivery_date": None,
            "shipping": ship("Alice Reader"),
        },
        {
            "order_id": "ORD-5002",
            "customer_name": "Alice Reader",
            "phone_number": "+1-555-0101",
            "book_name": "Night Letters",
            "book_type": "ebook",
            "status": "Delivered",
            "estimated_delivery_date": None,
            "delivery_date": d_delivered,
            "shipping": ship("Alice Reader"),
        },
        {
            "order_id": "ORD-5003",
            "customer_name": "Ben Audiophile",
            "phone_number": "+1-555-0102",
            "book_name": "Sound of Pages",
            "book_type": "audiobook",
            "status": "Order Processing",
            "estimated_delivery_date": d2,
            "delivery_date": None,
            "shipping": {
                "line1": "88 Sound Ave",
                "city": "Austin",
                "region": "TX",
                "postal_code": "78701",
                "country": "US",
                "label_name": "Ben Audiophile",
            },
        },
        {
            "order_id": "ORD-5004",
            "customer_name": "Clara Hardcover",
            "phone_number": "+1-555-0103",
            "book_name": "Marble & Ink",
            "book_type": "hardcover",
            "status": "Delivered",
            "estimated_delivery_date": None,
            "delivery_date": d_delivered,
            "shipping": {
                "line1": "400 Library Ln",
                "city": "Boston",
                "region": "MA",
                "postal_code": "02108",
                "country": "US",
                "label_name": "Clara Hardcover",
            },
        },
        {
            "order_id": "ORD-5005",
            "customer_name": "Dev Ebook",
            "phone_number": "+1-555-0104",
            "book_name": "Zero Latency",
            "book_type": "ebook",
            "status": "Delivered",
            "estimated_delivery_date": None,
            "delivery_date": d_delivered,
            "shipping": {
                "line1": "9 Pixel Rd",
                "line2": "Suite 200",
                "city": "Denver",
                "region": "CO",
                "postal_code": "80202",
                "country": "US",
                "label_name": "Dev Ebook",
            },
        },
        {
            "order_id": "ORD-5006",
            "customer_name": "Eve Mixed",
            "phone_number": "+1-555-0105",
            "book_name": "Paper Trails",
            "book_type": "paperback",
            "status": "In Transit",
            "estimated_delivery_date": d1,
            "delivery_date": None,
            "shipping": {
                "line1": "77 Chapter Ct",
                "city": "Portland",
                "region": "OR",
                "postal_code": "97201",
                "country": "US",
                "label_name": "Eve Mixed",
            },
        },
        {
            "order_id": "ORD-5007",
            "customer_name": "Eve Mixed",
            "phone_number": "+1-555-0105",
            "book_name": "Index of Stars",
            "book_type": "hardcover",
            "status": "Order Processing",
            "estimated_delivery_date": d2,
            "delivery_date": None,
            "shipping": {
                "line1": "77 Chapter Ct",
                "city": "Portland",
                "region": "OR",
                "postal_code": "97201",
                "country": "US",
                "label_name": "Eve Mixed",
            },
        },
        {
            "order_id": "ORD-5008",
            "customer_name": "Ben Audiophile",
            "phone_number": "+1-555-0102",
            "book_name": "Whispering Shelves",
            "book_type": "audiobook",
            "status": "Delivered",
            "estimated_delivery_date": None,
            "delivery_date": d_delivered,
            "shipping": {
                "line1": "88 Sound Ave",
                "city": "Austin",
                "region": "TX",
                "postal_code": "78701",
                "country": "US",
                "label_name": "Ben Audiophile",
            },
        },
        {
            "order_id": "ORD-5009",
            "customer_name": "Clara Hardcover",
            "phone_number": "+1-555-0103",
            "book_name": "Bound in Blue",
            "book_type": "hardcover",
            "status": "In Transit",
            "estimated_delivery_date": d1,
            "delivery_date": None,
            "shipping": {
                "line1": "400 Library Ln",
                "city": "Boston",
                "region": "MA",
                "postal_code": "02108",
                "country": "US",
                "label_name": "Clara Hardcover",
            },
        },
        {
            "order_id": "ORD-5010",
            "customer_name": "Dev Ebook",
            "phone_number": "+1-555-0104",
            "book_name": "Cached Thoughts",
            "book_type": "ebook",
            "status": "Order Processing",
            "estimated_delivery_date": d2,
            "delivery_date": None,
            "shipping": {
                "line1": "9 Pixel Rd",
                "line2": "Suite 200",
                "city": "Denver",
                "region": "CO",
                "postal_code": "80202",
                "country": "US",
                "label_name": "Dev Ebook",
            },
        },
    ]
    collection.delete_many({})
    collection.insert_many(orders)
    print(f"Inserted {len(orders)} orders.")


def _policy_documents() -> List[Document]:
    docs: List[Document] = []
    chunk_index = 0

    def add_docs(policy_type: str, title: str, chunks: List[str]) -> None:
        nonlocal chunk_index
        for i, text in enumerate(chunks):
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "policy_type": policy_type,
                        "title": title,
                        "chunk_index": i,
                        "chunk_id": chunk_index,
                    },
                )
            )
            chunk_index += 1

    add_docs("shipping", "Shipping Policy", SHIPPING_POLICY_CHUNKS)
    add_docs("return", "Return and Refund Policy", RETURN_POLICY_CHUNKS)
    add_docs("password-reset", "Password Reset Guide", PASSWORD_RESET_CHUNKS)
    return docs


def _seed_policies_vector(
    policies_coll,
    embeddings: VoyageAIEmbeddings,
) -> None:
    policies_coll.delete_many({})

    vstore = MongoDBAtlasVectorSearch(
        collection=policies_coll,
        embedding=embeddings,
        index_name=config.VECTOR_INDEX_NAME,
        text_key="text",
        embedding_key="embedding",
        relevance_score_fn="cosine",
        dimensions=config.VOYAGE_EMBED_DIMENSIONS,
        auto_create_index=True,
        auto_index_timeout=120,
    )

    documents = _policy_documents()
    vstore.add_documents(documents)
    print(f"Inserted {len(documents)} policy chunks into '{config.POLICIES_COLLECTION}'.")


def main() -> None:
    _require_env("MONGODB_URI")
    _require_env("VOYAGE_API_KEY")

    database = db.get_db()

    _seed_customers(database[config.CUSTOMER_COLLECTION])
    _seed_orders(database[config.ORDERS_COLLECTION])

    embeddings = VoyageAIEmbeddings()
    _seed_policies_vector(database[config.POLICIES_COLLECTION], embeddings)

    print("Done. Ensure your Atlas Vector Search index exists on policies.embedding")
    print(f"with filter field policy_type (index name: {config.VECTOR_INDEX_NAME}).")


if __name__ == "__main__":
    main()
