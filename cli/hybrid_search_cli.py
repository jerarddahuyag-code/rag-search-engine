import argparse

from lib.evaluation_commands import evaluate_results
from lib.utils import DEFAULT_ALPHA, DEFAULT_K, DEFAULT_SEARCH_LIMIT
from lib.hybrid_search_commands import min_max_normalization, rrf_search_command, weighted_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores using min-max normalization")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="A list of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Gets the weighted score of a document's keyword and semantic score based on a query. ")
    weighted_search_parser.add_argument("query", type=str, help="The query to search for")
    weighted_search_parser.add_argument("--alpha", nargs="?", default=DEFAULT_ALPHA, type=float, help="The balance between keyword and semantic score")
    weighted_search_parser.add_argument("--limit", nargs="?", default=DEFAULT_SEARCH_LIMIT, type=int, help="Number of movies to return")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Gets the weighted score of a document's keyword and semantic score based on a query. ")
    rrf_search_parser.add_argument("query", type=str, help="The query to search for")
    rrf_search_parser.add_argument("--k", nargs="?", default=DEFAULT_K, type=float, help="The value that dictates how much more weight we give to higher vs lower ranked results")
    rrf_search_parser.add_argument("--limit", nargs="?", default=DEFAULT_SEARCH_LIMIT, type=int, help="Number of movies to return")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Method to use for reranking search results")
    rrf_search_parser.add_argument("--evaluate", action='store_true', help="Print out an evaluation of the search results from an LLM")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            norm_scores = min_max_normalization(args.scores)
            for s in norm_scores:
                print(f"* {s:.4f}")
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            results = rrf_search_command(args.query, args.k, args.enhance, args.rerank_method, args.limit)
            print("====================== Final Results ======================")
            for i, result in enumerate(results):
                print(f"{i + 1}. {result['movie']['title']}")
                if args.rerank_method:
                    print(f"Re-rank score ({args.rerank_method}): {result['rerank_score']}")
                print(f"RRF Score: {result['hybrid_score']}")
                print(f"BM25: {result['keyword_score']} Semantic: {result['semantic_score']}")
                print(f"{result['movie']['description']}")
            if args.evaluate:
                evaluate_results(args.query, results, args.rerank_method)
        case _:
            parser.print_help()
if __name__ == "__main__":
    main()