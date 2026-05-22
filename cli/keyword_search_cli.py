#!/usr/bin/env python3

import argparse
from lib.search_command import search

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies = search(args.query)
            for i, movie in enumerate(movies):
                print(f"{i + 1}. {movie['title']}")
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()