import argparse
from tokenizing import Tokenizer
from embedding import Embedding
from LSTM import LSTMModeling
from LR import LRModeling
from CB import CBModeling


def get_args_parser():
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument("--data", type=str, default="data/sample_post.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_trials", type=int, default=10)
    ######################
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--corpus", type=str)
    parser.add_argument("--label", type=str)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    preprocessor = Tokenizer(args)
    res = preprocessor.tokenizing()
    for tokenized_sentences, name in res:
        embedding = Embedding(args, tokenized_sentences)
        w2v_model = embedding.get_embedding_model()
        for model in ["LSTM", "LR", "CatBoost"]:
            if model == "LSTM":
                modeling = LSTMModeling(args, embedding, w2v_model, name)
                modeling.train_and_evaluate(epochs=args.num_epochs)
            elif model == "LR":
                modeling = LRModeling(args, embedding, w2v_model, name)
                modeling.train_and_evaluate(epochs=args.num_epochs)
            elif model == "CatBoost":
                modeling = CBModeling(args, embedding, w2v_model, name)
                modeling.train_and_evaluate(epochs=args.num_epochs)
    print("----------  Done  ----------")
