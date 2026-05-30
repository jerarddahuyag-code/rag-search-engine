import argparse

from lib.utils import DEFAULT_ALPHA, DEFAULT_SEARCH_LIMIT
from lib.hybrid_search_commands import min_max_normalization, weighted_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores using min-max normalization")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="A list of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Gets the weighted score of a document's keyword and semantic score based on a query. ")
    weighted_search_parser.add_argument("query", type=str, help="The query to search for")
    weighted_search_parser.add_argument("--alpha", nargs="?", default=DEFAULT_ALPHA, type=float, help="The balance between keyword and semantic score")
    weighted_search_parser.add_argument("--limit", nargs="?", default=DEFAULT_SEARCH_LIMIT, type=int, help="Number of movies to return")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm_scores = min_max_normalization(args.scores)
            for s in norm_scores:
                print(f"* {s:.4f}")
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()
if __name__ == "__main__":
    main()