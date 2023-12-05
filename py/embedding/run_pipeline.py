import argparse
from tokenizing import Tokenizer
from embedding import Embedding
from base_modeling import TextModel
import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument("--data", type=str, default="data/sample_comment.csv")
    parser.add_argument("--label", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sp_option", type=str, default=False)
    parser.add_argument("--corpus", type=str)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    preprocessor = Tokenizer(args)
    res = preprocessor.tokenizing()

    for tokenized_sentences in res:
        embedding = Embedding(args, tokenized_sentences)
        w2v_model = embedding.get_embedding_model()
        # print(w2v_model.wv)
        textmodel = TextModel(args, w2v_model)
        textmodel.train_model()

# import argparse
# from tokenizing import Tokenizer
# from embedding import Embedding
# # from base_modeling import TextModel


# def get_args_parser():
#     parser = argparse.ArgumentParser(description="Run the pipeline")
#     parser.add_argument("--data", type=str, default="data/sample_comment.csv")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--device", type=str, default="cuda")
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--num_epochs", type=int, default=10)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--sp_option", type=str, default=False)
#     return parser


# if __name__ == "__main__":
#     parser = get_args_parser()
#     args = parser.parse_args()

#     preprocessor = Tokenizer(args)
#     res = preprocessor.tokenizing()

#     for tokenized_sentences in res:
#         embedding = Embedding(args, tokenized_sentences)
#         word_vectors = embedding.get_embedding_vector()
