#!/usr/bin/env python3
"""Small dependency-free TF-IDF retrieval lab for Lecture 4.

The documents are fictional course fixtures. The script intentionally keeps
retrieval and metadata filtering visible so students can modify one variable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def tfidf_vectors(texts: list[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    tokenised = [tokens(text) for text in texts]
    document_frequency: Counter[str] = Counter()
    for doc_tokens in tokenised:
        document_frequency.update(set(doc_tokens))

    n_documents = len(texts)
    inverse_document_frequency = {
        term: math.log((1 + n_documents) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }

    vectors: list[dict[str, float]] = []
    for doc_tokens in tokenised:
        counts = Counter(doc_tokens)
        vectors.append(
            {
                term: count * inverse_document_frequency[term]
                for term, count in counts.items()
            }
        )
    return vectors, inverse_document_frequency


def query_vector(text: str, inverse_document_frequency: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens(text))
    unseen_idf = math.log(1 + len(inverse_document_frequency)) + 1
    return {
        term: count * inverse_document_frequency.get(term, unseen_idf)
        for term, count in counts.items()
    }


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="Which GPU option is currently approved for a 900 GPU-hour monthly workload?",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Filter out superseded, expired, and draft records before ranking.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    corpus_path = Path(__file__).with_name("corpus.json")
    documents = json.loads(corpus_path.read_text(encoding="utf-8"))
    if args.current_only:
        documents = [doc for doc in documents if doc["status"] == "current"]

    texts = [f'{doc["title"]} {doc["text"]}' for doc in documents]
    document_vectors, inverse_document_frequency = tfidf_vectors(texts)
    vector = query_vector(args.query, inverse_document_frequency)

    ranked = []
    for document, document_vector in zip(documents, document_vectors):
        ranked.append(
            {
                "document_id": document["document_id"],
                "score": round(cosine(vector, document_vector), 4),
                "effective_date": document["effective_date"],
                "status": document["status"],
                "authority": document["authority"],
                "title": document["title"],
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    result = {
        "query": args.query,
        "top_k": args.top_k,
        "current_only": args.current_only,
        "results": ranked[: args.top_k],
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
