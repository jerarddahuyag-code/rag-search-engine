#!/usr/bin/env python3

import argparse
from lib.keyword_search_commands import search, build

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds the inverted index for the movies")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies = search(args.query)
            for i, movie in enumerate(movies):
                print(f"{i + 1}. {movie['title']}")
            pass
        case "build":
            print("Building inverted index")
            build()
            print("Successfully build inverted index")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()