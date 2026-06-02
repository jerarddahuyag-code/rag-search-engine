import argparse

from lib.utils import DEFAULT_K, DEFAULT_SEARCH_LIMIT
from lib.hybrid_search_commands import rrf_search_command
from lib.rag_commands import citation_command, question_command, rag_command, summarize_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize the results from search for the user"
    )
    summarize_parser.add_argument("query", type=str, help="Query to search for in the database")
    summarize_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of movies to search for")

    citations_parser = subparsers.add_parser(
        "citations", help="Summarize results from search and add citations to it"
    )
    citations_parser.add_argument("query", type=str, help="Query to search for in the database")
    citations_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of movies to search for")  

    citations_parser = subparsers.add_parser(
        "question", help="Answer a question based on search results"
    )
    citations_parser.add_argument("question", type=str, help="Question to query and answer")
    citations_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of movies to search for")
  

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rrf_search_command(query, DEFAULT_K, "spell", "batch")
            response = rag_command(query, results)
            print("Search Results:")
            for result in results:
                print(f"- {result["movie"]["title"]}")
            print("RAG Response")
            print(response)
        case "summarize":
            query = args.query
            results = rrf_search_command(query, DEFAULT_K, "spell", "batch", args.limit)
            response = summarize_command(query, results)
            print("Search Results:")
            for result in results:
                print(f"- {result["movie"]["title"]}")
            print("LLM Summary")
            print(response)
        case "citations":
            query = args.query
            results = rrf_search_command(query, DEFAULT_K, "rewrite", "batch", args.limit)
            response = citation_command(query, results)
            print("Search Results:")
            for result in results:
                print(f"- {result["movie"]["title"]}")
            print("LLM Answer")
            print(response)
        case "question":
            query = args.question
            results = rrf_search_command(query, DEFAULT_K, "rewrite", "batch", args.limit)
            response = question_command(query, results)
            print("Search Results:")
            for result in results:
                print(f"- {result["movie"]["title"]}")
            print("LLM Answer")
            print(response)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()