import argparse

from lib.multimodal_search_commands import image_search_command
from lib.multimodal_search_commands import verify_image_embedding_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Generate embedding from image"
    )
    verify_parser.add_argument("image", type=str, help="Path to image to embed")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search database using an image"
    )
    image_search_parser.add_argument("image", type=str, help="Path to image to use for search")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding_command(args.image)
        case "image_search":
            image_search_command(args.image)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()