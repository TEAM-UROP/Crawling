import pandas as pd
from gensim.models import Word2Vec
import optuna
from sklearn.model_selection import train_test_split

class Embedding:
    def __init__(self, corpus):
        self.df = pd.read_csv(corpus)
        self.corpus = [
            str(sentence).lower().split()
            for sentence in self.df["comments"]
            if pd.notnull(sentence)
        ]

    def get_split_data(self):
        train_corpus, temp_corpus = train_test_split(self.corpus, test_size=0.5, random_state=42)
        test_corpus, validation_corpus = train_test_split(temp_corpus, test_size=0.4, random_state=42)
        return train_corpus, test_corpus, validation_corpus

    def objective(self, trial, corpus):
        vector_size = trial.suggest_int("vector_size", 10, 100)
        window = trial.suggest_int("window", 3, 10)
        min_count = trial.suggest_int("min_count", 5, 30)
        sg = trial.suggest_categorical("sg", [0])

        model = Word2Vec(
            vector_size=vector_size, window=window, min_count=min_count, sg=sg
        )
        model.build_vocab(corpus)
        model.train(
            corpus,
            total_examples=model.corpus_count,
            epochs=model.epochs,
            compute_loss=True,
        )

        loss = model.get_latest_training_loss()
        return loss

    def optimize_hyperparameters(self, corpus):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, corpus), n_trials=100)
        return study.best_params

    def train_word2vec_model(self, corpus, best_params):
        model = Word2Vec(
            vector_size=best_params["vector_size"],
            window=best_params["window"],
            min_count=best_params["min_count"],
            sg=best_params["sg"],
        )
        model.build_vocab(corpus)
        model.train(
            corpus,
            total_examples=model.corpus_count,
            epochs=model.epochs,
            compute_loss=True,
        )
        return model

    def get_embedding_vector(self):
        train_corpus, test_corpus, validation_corpus = self.get_split_data()

        best_params_train = self.optimize_hyperparameters(train_corpus)
        best_params_test = self.optimize_hyperparameters(test_corpus)
        best_params_validation = self.optimize_hyperparameters(validation_corpus)

        best_model_train = self.train_word2vec_model(train_corpus, best_params_train)
        best_model_test = self.train_word2vec_model(test_corpus, best_params_test)
        best_model_validation = self.train_word2vec_model(validation_corpus, best_params_validation)

        return best_model_train, best_model_test, best_model_validation

if __name__ == "__main__":
    embedding = Embedding("./tokenized_0.csv")

    best_model_train, best_model_test, best_model_validation = embedding.get_embedding_vector()
    # 각 데이터셋에 대한 Word2Vec 모델(best_model_train, best_model_test, best_model_validation)을 사용하여 후속 작업 수행
