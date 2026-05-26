#!/usr/bin/env python3

import argparse

from lib.semantic_search_commands import verify_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    subparser.add_parser("verify", help="Prints out model information used for semantic search")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()