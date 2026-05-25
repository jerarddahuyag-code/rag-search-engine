#!/usr/bin/env python3

import argparse
from lib.keyword_search_commands import (
    search,
    build,
    tf, 
    idf, 
    tf_idf
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds the inverted index for the movies")

    tf_parser = subparsers.add_parser("tf", help="Get the frequency of the term in the specified document id")
    tf_parser.add_argument("id", type=int, help="Document Id to search for term frequency")
    tf_parser.add_argument("term", type=str, help="Term to count")

    idf_parser = subparsers.add_parser("idf", help="Gets the inverse document frequency of a given term")
    idf_parser.add_argument("term", type=str, help="Term to get inverse document frequency of")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate tf-idf of a term in a specified document")
    tfidf_parser.add_argument("id", type=int, help="Document Id to calculate term tf-idf")
    tfidf_parser.add_argument("term", type=str, help="Term to calculate for tf-idf")

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
            print("Successfully built inverted index")
        case "tf":
            print(f"Getting frequency of: {args.term} in movie with Id: {args.id}")
            count = tf(args.id, args.term)
            print(count)
        case "idf":
            score = idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {score:.2f}")
        case "tfidf":
            value = tf_idf(args.id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {value:.2f}")
            print
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()