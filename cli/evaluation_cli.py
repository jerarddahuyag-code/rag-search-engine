import argparse
import json

from lib.utils import (
        DEFAULT_K, 
        GOLDEN_DATASET_PATH
    )
from lib.hybrid_search_commands import rrf_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
         "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open(GOLDEN_DATASET_PATH, "r") as f:
        data = json.load(f)
    test_cases = data['test_cases']
    print(f"k={limit}")
    for case in test_cases:
        query = case['query']
        expected = case['relevant_docs']
        search_results = rrf_search_command(query, DEFAULT_K, limit=limit)
        correct_count = 0
        retrieved_titles = []
        for res in search_results:
            doc = res['movie']
            retrieved_titles.append(doc['title'])
            if doc['title'] in expected:
                correct_count = correct_count + 1
        precision = correct_count / len(search_results)
        recall = correct_count / len(expected)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {query}")
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Recall@{limit}: {recall:.4f}")
        print(f"    - F1 Score: {f1:.4f}")
        print(f"    - Retrieved: {", ".join(retrieved_titles)}")
        print(f"    - Relevant: {", ".join(expected)}")

if __name__ == "__main__":
     main()