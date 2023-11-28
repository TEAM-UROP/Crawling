import argparse
from tokenizing import Tokenizer
from embedding import Embedding
from base_modeling import TextModel


def get_args_parser():
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/sample_comment.csv",
        help="Path to the data folder",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the code"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    preprocessor = Tokenizer(args)
    res = preprocessor.tokenizing()

    for tokenized_sentences in res:
        embedding = Embedding(args, tokenized_sentences)
        word_vectors = embedding.get_embedding_vector()
