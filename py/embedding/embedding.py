from gensim.models import Word2Vec
import optuna


class Embedding:
    def __init__(self, args, sereis):
        self.corpus = []
        for i in sereis.values:
            # temp = []
            # print(i)
            self.corpus.append(i.split(" "))
        args.corpus = self.corpus

    def objective(self, trial):
        vector_size = trial.suggest_int("vector_size", 10, 100)
        window = trial.suggest_int("window", 3, 10)
        min_count = trial.suggest_int("min_count", 5, 30)
        sg = trial.suggest_categorical("sg", [0])

        model = Word2Vec(
            vector_size=vector_size, window=window, min_count=min_count, sg=sg
        )
        model.build_vocab(self.corpus)
        model.train(
            self.corpus,
            total_examples=model.corpus_count,
            epochs=model.epochs,
            compute_loss=True,
        )

        loss = model.get_latest_training_loss()
        return loss

    def optimize_hyperparameters(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=1)
        return study.best_params

    def train_word2vec_model(self, best_params):
        model = Word2Vec(
            vector_size=best_params["vector_size"],
            window=best_params["window"],
            # min_count=best_params["min_count"],
            min_count=5,
            sg=best_params["sg"],
        )
        model.build_vocab(self.corpus)
        model.train(
            self.corpus,
            total_examples=model.corpus_count,
            epochs=model.epochs,
            compute_loss=True,
        )
        return model

    def get_embedding_model(self):
        best_params = self.optimize_hyperparameters()
        word2vec_model = self.train_word2vec_model(best_params)
        return word2vec_model


if __name__ == "__main__":
    embedding = Embedding("./tokenized_0.csv")
    word2vec_model = embedding.get_embedding_model()
