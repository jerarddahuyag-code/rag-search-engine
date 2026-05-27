#!/usr/bin/env python3

import argparse

from lib.utils import DEFAULT_CHUNK_SIZE, DEFAULT_SEARCH_LIMIT
from lib.semantic_search_commands import (
    chunk_command,
    embed_query,
    search_command,
    verify_embeddings,
    verify_model,
    embed_text
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Prints out model information used for semantic search")

    embedding_parser = subparsers.add_parser("embed_text", help="Prints the embedded version of input text")
    embedding_parser.add_argument("text", type=str, help="The text to embed")
   
    subparsers.add_parser("verify_embeddings", help="Verifies and prints the cached embeddings")

    query_embedding_parser = subparsers.add_parser("embed_query", help="Prints embedded version of input query")
    query_embedding_parser.add_argument("query", type=str, help="The query to embed")

    semantic_search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    semantic_search_parser.add_argument("query", type=str, help="Search query")
    semantic_search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit of returned relevant documents")
    
    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into specified word size")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=DEFAULT_CHUNK_SIZE, help="Chunk size in words")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()